import os
import yaml
import json

import pandas as pd
import numpy as np

from src.data_pipeline.soil.utils_soil.default_soil_variables import (
    default_som_content,
    default_range_pf_values,
    default_pf_field_capacity,
    default_pf_wilting_point,
    default_surface_conductivity,
)



def calculate_is_topsoil(zmin, zmax, zmax_topsoil):
    if (zmin <= zmax_topsoil) & (zmax <= zmax_topsoil):
        is_topsoil = True
    else:
        is_topsoil = False
    return is_topsoil


def calculate_van_genuchten(
        df_soilgrids: pd.DataFrame,
        lower_boundary_top_soil: float = 100.,
    ) -> pd.DataFrame:

    f_C_to_OM = default_som_content()
    pml_to_pct = 0.1

    ptfw = PedotransferFunctionsWosten()
    df_vgp = df_soilgrids.copy()[["latitude", "longitude", "zmin", "zmax", "soc", "phh2o", "nitrogen"]]
    df_vgp["C"] = df_soilgrids.clay.copy()
    df_vgp["D"] = df_soilgrids.bdod.copy()
    df_vgp["S"] = df_soilgrids.silt.copy()
    df_vgp["OM"] = df_soilgrids.soc.copy() * pml_to_pct * f_C_to_OM
    # assume low residual soil moisture content
    theta_r = 0.01

    alphas = []
    ns = []
    labdas = []
    k_sats = []
    theta_rs = []
    theta_ss = []

    for i in range(len(df_soilgrids)):
        is_topsoil = calculate_is_topsoil(df_vgp.zmin.iloc[i], df_vgp.zmax.iloc[i], lower_boundary_top_soil)
        vg_dict = ptfw.calculate_van_genuchten_parameters(
            df_vgp.C.iloc[i], df_vgp.D.iloc[i], df_vgp.S.iloc[i], df_vgp.OM.iloc[i], theta_r, is_topsoil
        )
        alphas.append(vg_dict["alpha"])
        ns.append(vg_dict["n"])
        labdas.append(vg_dict["lambda"])
        k_sats.append(vg_dict["k_sat"])
        theta_rs.append(vg_dict["theta_r"])
        theta_ss.append(vg_dict["theta_s"])

    df_vgp["alpha"] = alphas
    df_vgp["n"] = ns
    df_vgp["labda"] = labdas
    df_vgp["k_sat"] = k_sats
    df_vgp["theta_r"] = theta_rs
    df_vgp["theta_s"] = theta_ss

    return df_vgp


def generate_df_soil_input(
        df_vgp: pd.DataFrame,
) -> pd.DataFrame:

    pct_to_frac = 0.01

    df_model_input = pd.DataFrame()
    df_model_input["Thickness"] = df_vgp.zmax.copy() - df_vgp.zmin.copy()

    # Soil bulk density, default unit is cg / cm³ = 10 kg / m³
    df_model_input["RHOD"] = df_vgp.D.copy() * 10
    df_model_input["Soil_pH"] = df_vgp.phh2o.copy()
    df_model_input["FSOMI"] = df_vgp.OM.copy() * pct_to_frac
    df_model_input["CNRatioSOMI"] = df_vgp.soc.copy() / df_vgp.nitrogen.copy()

    # assume value for critical air content
    df_model_input["CRAIRC"] = 0.03

    # determine range of pF values
    pFs = default_range_pf_values()

    CONDfromPF_perlayer = []
    SMfromPF_perlayer = []

    for i in range(len(df_vgp)):
        CONDfromPF = []
        SMfromPF = []
        for j, pF in enumerate(pFs):
            r = calculate_soil_moisture_content(
                pF,
                df_vgp.alpha.iloc[i],
                df_vgp.n.iloc[i],
                df_vgp.theta_r.iloc[i],
                df_vgp.theta_s.iloc[i],
            )
            SMfromPF.extend([pF, float(r)])
            r = calculate_log10_hydraulic_conductivity(
                pF,
                df_vgp.alpha.iloc[i],
                df_vgp.labda.iloc[i],
                df_vgp.k_sat.iloc[i],
                df_vgp.n.iloc[i],
            )
            CONDfromPF.extend([pF, float(r)])
        CONDfromPF_perlayer.append(CONDfromPF)
        SMfromPF_perlayer.append(SMfromPF)

    df_model_input["CONDfromPF"] = CONDfromPF_perlayer
    df_model_input["SMfromPF"] = SMfromPF_perlayer

    return df_model_input


def generate_soil_yaml(
        df_model_input: pd.DataFrame,
        filename: str = None,
) -> str:
    PFFieldCapacity = default_pf_field_capacity()
    PFWiltingPoint = default_pf_wilting_point()
    SurfaceConductivity = default_surface_conductivity()

    nlayers = len(df_model_input)
    Thickness = df_model_input.Thickness.to_list()
    CNRatioSOMI = df_model_input.CNRatioSOMI.to_list()
    CRAIRC = df_model_input.CRAIRC.to_list()
    FSOMI = df_model_input.FSOMI.to_list()
    RHOD = df_model_input.RHOD.to_list()
    Soil_pH = df_model_input.Soil_pH.to_list()
    SMfromPF_perlayer = df_model_input.SMfromPF.to_list()
    CONDfromPF_perlayer = df_model_input.CONDfromPF.to_list()

    # below we generate the header of the soil input file as YAML input structure
    soil_input_yaml = f"""
    RDMSOL: {sum(Thickness)}
    SoilProfileDescription:
        PFWiltingPoint: {PFWiltingPoint}
        PFFieldCapacity: {PFFieldCapacity}
        SurfaceConductivity: {SurfaceConductivity}
        GroundWater: false
        SoilLayers:
    """

    # Here we generate the properties for each soil layer including layer thickness, hydraulic properties,
    # organic matter content, etc.
    for i in range(nlayers):
        s = f"""    - Thickness: {Thickness[i]}
          CNRatioSOMI: {CNRatioSOMI[i]}
          CRAIRC: {CRAIRC[i]}
          FSOMI: {FSOMI[i]}
          RHOD: {RHOD[i]}
          Soil_pH: {Soil_pH[i]}
          SMfromPF: {make_string_table(SMfromPF_perlayer[i])}
          CONDfromPF: {make_string_table(CONDfromPF_perlayer[i])}
    """
        soil_input_yaml += s

    # A SubSoilType needs to be defined. In this case we make the subsoil equal to the properties
    # of the deepest soil layer.
    soil_input_yaml += \
        f"""    SubSoilType:
          CNRatioSOMI: {CNRatioSOMI[-1]}
          CRAIRC: {CRAIRC[-1]}
          FSOMI: {FSOMI[-1]}
          RHOD: {RHOD[-1]}
          Soil_pH: {Soil_pH[-1]}
          Thickness: {Thickness[-1]}
          SMfromPF: {make_string_table(SMfromPF_perlayer[-1])}
          CONDfromPF: {make_string_table(CONDfromPF_perlayer[-1])}
    """

    soil_dict = yaml.safe_load(soil_input_yaml)
    soil_yaml = json.dumps(soil_dict, indent=6)
    return soil_yaml


def dump_soil_yaml(soil_input_yaml, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(soil_input_yaml)


class PedotransferFunctionsWosten():
    """
    Taken from https://github.com/ajwdewit/pcse_notebooks/
    """
    def calculate_van_genuchten_parameters(self, C, D, S, OM, theta_r, topSoil):
        if(OM == 0):
            OM = 0.01
        dict_vg = {}
        dict_vg["alpha"] = self.calculate_alpha(C, D, S, OM, topSoil)
        dict_vg["n"] = self.calculate_n(C, D, S, OM, topSoil)
        dict_vg["lambda"] = self.calculate_lambda(C, D, S, OM, topSoil)
        dict_vg["k_sat"] = self.calculate_k_sat(C, D, S, OM, topSoil)
        dict_vg["theta_r"] = theta_r
        dict_vg["theta_s"] = self.calculate_theta_s(C, D, S, OM, topSoil)
        return dict_vg

    def calculate_alpha(self, C, D, S, OM, topSoil):
        t_alpha = self.calculate_transformed_alpha(C, D, S, OM, topSoil)
        alpha = np.exp(t_alpha)
        return alpha

    def calculate_n(self, C, D, S, OM, topSoil):
        t_n = self.calculate_transformed_n(C, D, S, OM, topSoil)
        n = np.exp(t_n) + 1
        return n

    def calculate_lambda(self, C, D, S, OM, topSoil):
        t_labda = self.calculate_transformed_lambda(C, D, S, OM, topSoil)
        labda = (10 * np.exp(t_labda) - 10) / (1 + np.exp(t_labda))
        return labda

    def calculate_k_sat(self, C, D, S, OM, topSoil):
        t_k_sat = self.calculate_transformed_ksat(C, D, S, OM, topSoil)
        k_sat = np.exp(t_k_sat)
        return k_sat

    def calculate_theta_s(self, C, D, S, OM, topSoil):
        if(topSoil):
            TS = 1.
        else:
            TS = 0.
        theta_s = 0.7919 + 0.001691 * C - 0.29619 * D - 0.000001491 * S * S + 0.0000821 * OM * OM + 0.02427 * (1 / C) + 0.01113 * (1 / S) + \
                0.01472 * np.log(S) - 0.0000733 * OM * C - 0.000619 * D * C - 0.001183 * D * OM - 0.0001664 * topSoil * S
        return theta_s

    def calculate_transformed_alpha(self, C, D, S, OM, topSoil):
        if(topSoil):
            TS = 1.
        else:
            TS = 0.
        t_alpha = -14.96 + 0.03135 * C + 0.0351 * S + 0.646 * OM + 15.29 * D - 0.192 * topSoil - 4.671 * D * D - 0.000781 * C * C - \
                0.00687 * OM * OM + 0.0449 * (1 / OM) + 0.0663 * np.log(S) + 0.1482 * np.log(OM) - 0.04546 * D * S - 0.4852 * D * OM + 0.00673 * topSoil * C
        return t_alpha

    def calculate_transformed_n(self, C, D, S, OM, topSoil):
        if(topSoil):
            TS = 1.
        else:
            TS = 0.
        t_n = -25.23 - 0.02195 * C + 0.0074 * S - 0.1940 * OM + 45.5 * D - 7.24 * D * D + 0.0003658 * C * C + 0.002885 * OM * OM - 12.81 * (1 / D) - \
                0.1524 * (1 / S) - 0.01958 * (1 / OM) - 0.2876 * np.log(S) - 0.0709 * np.log(OM) - 44.6 * np.log(D) - 0.02264 * D * C + 0.0896 * D * OM + 0.00718 * topSoil * C
        return t_n

    def calculate_transformed_lambda(self, C, D, S, OM, topSoil):
        if(topSoil):
            TS = 1.
        else:
            TS = 0.
        t_lambda = 0.0202 + 0.0006193 * C * C - 0.001136 * OM * OM - 0.2316 * np.log(OM) - 0.03544 * D * C + 0.00283 * D * S + 0.0488 * D * OM
        return t_lambda

    def calculate_transformed_ksat(self, C, D, S, OM, topSoil):
        if(topSoil):
            TS = 1.
        else:
            TS = 0.
        t_k_sat = 7.755 + 0.0352 * S + 0.93 * topSoil - 0.967 * D * D - 0.000484 * C * C - 0.000322 * S * S + \
                0.001 * (1 / S) - 0.0748 * (1 / OM) - 0.643 * np.log(S) - 0.01398 * D * C - 0.1673 * D * OM + \
                0.02986 * topSoil * C - 0.03305 * topSoil * S
        return t_k_sat


def calculate_water_potential_form_pf(pF):
    psi = np.power(10, pF)
    return psi


def calculate_soil_moisture_content(pF, alpha, n, theta_r, theta_s):
    psi = calculate_water_potential_form_pf(pF)
    soil_moisture_content = theta_r + (theta_s - theta_r) / np.power(1 + (np.power(alpha * psi, n)), 1 - 1 / n)
    return soil_moisture_content


def calculate_log10_hydraulic_conductivity(pF, alpha, labda, k_sat, n):
    psi = calculate_water_potential_form_pf(pF)
    m = 1 - 1 / n
    ah = alpha * psi
    h1 = np.power(1 + np.power(ah, n), m)
    h2 = np.power(ah, n - 1)
    denom = np.power(1 + np.power(ah, n), m * (labda + 2))
    k_h = k_sat * np.power(h1 - h2, 2) / denom
    COND = np.log10(k_h)
    return COND


def make_string_table(XY_table):
    """Converts a list of X,Y pairs into a formatted string table.
    """
    s = "["
    for x, y in zip(XY_table[0::2], XY_table[1::2]):
        s += f"{x:4.1f}, {y:7.4f}, "
    s += "]"
    s = s.replace("  ", " ")
    return s
