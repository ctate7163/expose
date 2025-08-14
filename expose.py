import streamlit as st
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from copy import deepcopy
import os


# Load detector data from JSON file
def load_detectors_data():
    try:
        with open("detectors/detectors.jsonc", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ detectors/detectors.jsonc file not found. Please ensure the detectors folder contains the detector specifications.")
        st.stop()

# Load CSV data with error handling
def load_csv_data(filepath):
    try:
        return np.loadtxt(filepath, delimiter=',', skiprows=1)
    except FileNotFoundError:
        st.error(f"❌ CSV file not found: {filepath}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading CSV file {filepath}: {str(e)}")
        st.stop()

# Load predefined camera configurations
def load_camera_configs():
    try:
        with open("./cameras/cameras.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ cameras/cameras.json file not found. Please create the cameras folder and add camera configurations.")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading camera configurations: {str(e)}")
        return {}


class Camera:
    def __init__(self, detector_name, band='MONO', pixel_binning=False, wavelengths_nm=None):
        if wavelengths_nm is None:
            self.wavelengths_nm = np.arange(300, 1100, 1)
        else:
            self.wavelengths_nm = wavelengths_nm

        self.detector_name = detector_name
        self.detectors = load_detectors_data()
        
        if detector_name not in self.detectors:
            st.error(f"❌ Unknown detector: {detector_name}")
            st.stop()
        
        self.detector = self.detectors[detector_name].copy()
        self.detector['band'] = band
        
        # Load QE curve from CSV file
        qe_filename = self.detector['qe_curve_csv'] + '_' + band + '.csv'
        qe_filepath = f"./detectors/{qe_filename}"
        
        qe_data = load_csv_data(qe_filepath)
        wav = qe_data[:, 0]
        qe = qe_data[:, 1]
        self.qe_curve = np.interp(self.wavelengths_nm, wav, qe, left=0, right=0)
        
        self.pixel_binning = pixel_binning

    def set_lens(self, focal_length_mm, f_number, mean_transmission=0.9):
        self.lens_transmission = mean_transmission * np.ones_like(self.wavelengths_nm)
        self.lens_f_number = f_number
        self.lens_focal_length_mm = focal_length_mm
        self.lens_aperture_mm = focal_length_mm / f_number
        self.lens_mean_transmission = mean_transmission

        self.fov_H_deg = 2 * np.arctan((self.detector['pixel_size_um'] * 1e-6 * self.detector['pixel_count_H']) / 
                                       (2 * self.lens_focal_length_mm * 1e-3)) * 180 / np.pi
        self.fov_V_deg = 2 * np.arctan((self.detector['pixel_size_um'] * 1e-6 * self.detector['pixel_count_V']) / 
                                       (2 * self.lens_focal_length_mm * 1e-3)) * 180 / np.pi
        self.ifov_rad = (self.detector['pixel_size_um'] / 1e6) / (self.lens_focal_length_mm * 1e-3)
        
        # Diffraction limited resolution (2*1.22 * lambda / D) for 550nm
        wavelength_m = 550e-9  # 550 nm in meters
        aperture_diameter_m = self.lens_aperture_mm * 1e-3
        self.diffraction_limit_rad = 2*1.22 * wavelength_m / aperture_diameter_m

    def set_filter(self, cut_on_nm=400, cut_off_nm=1000, peak_transmission=0.95):
        self.filter_cut_on_nm = cut_on_nm
        self.filter_cut_off_nm = cut_off_nm
        self.filter_transmission_peak = peak_transmission
        
        self.filter_transmission = peak_transmission * np.ones_like(self.wavelengths_nm)
        self.filter_transmission[((self.wavelengths_nm < cut_on_nm) | 
                                 (self.wavelengths_nm > cut_off_nm))] = 0.0

    def set_target_solar_albedo_incidence(self, distance_AU=1.0, albedo=[0.05, 0.15], 
                                        incidence_angle_deg=0.0, optical_depth=0.0, target_name=""):
        # Load solar irradiance data from CSV
        solar_data = load_csv_data("targets/solar_e490_00a_amo.csv")
        wav = solar_data[:, 0]
        rad = solar_data[:, 1] * 1e-3  # Convert from W/m2/micron to W/m2/nm
        solar_radiance_1AU = np.interp(self.wavelengths_nm, wav, rad, left=0, right=0)

        def albedoSpectrum(v_, r_, l=self.wavelengths_nm):
            l_ = [300, 450, 575, 1100]
            albedo_ = [v_, v_, r_, r_]
            albedo = np.interp(l, l_, albedo_)
            return albedo
        
        def albedoSpectrum(v_, r_, l=self.wavelengths_nm):
            l_ = [300, 400, 1100]
            albedo_ = [v_, v_, r_]
            albedo = np.interp(l, l_, albedo_)
            return albedo

        if isinstance(albedo, list):
            self.target_albedo_blue = albedo[0]
            self.target_albedo_red = albedo[1]
            self.target_albedo = albedoSpectrum(albedo[0], albedo[1])
        else:
            self.target_albedo_blue = albedo
            self.target_albedo_red = albedo
            self.target_albedo = albedoSpectrum(albedo, albedo)

        self.distance_AU = distance_AU
        self.target_name = target_name
        self.incidence_angle_deg = incidence_angle_deg

        self.target_irradiance = (1/distance_AU)**2 * solar_radiance_1AU * np.cos(np.radians(incidence_angle_deg))
        self.target_exitance = self.target_irradiance * self.target_albedo
        self.target_radiance = self.target_exitance / np.pi

        if optical_depth == 0:
            self.atmospheric_correction_factor = 1.0
        else:
            self.atmospheric_correction(incidence_angle_deg, optical_depth)

    def atmospheric_correction(self, incidence_angle_deg=0.0, optical_depth=0.0):
        self.atmospheric_correction_optical_depth = optical_depth
        if incidence_angle_deg < 0 or incidence_angle_deg > 90:
            raise ValueError(f"Invalid incidence angle: {incidence_angle_deg} degrees. Must be between 0 and 90 degrees.")
        self.atmospheric_correction_incidence_angle_deg = incidence_angle_deg

        self.atmospheric_correction_factor = np.exp( -optical_depth / 6 / np.cos(np.radians(incidence_angle_deg)))
        
        self.target_irradiance *= self.atmospheric_correction_factor
        self.target_exitance *= self.atmospheric_correction_factor
        self.target_radiance *= self.atmospheric_correction_factor

    def calculate_electron_rate(self):
        hc_Jm = 6.62607015e-34 * 2.99792458e8
        wavelength_step_nm = self.wavelengths_nm[1] - self.wavelengths_nm[0]

        effective_area_m2 = (np.pi/4) * (1e-6 * self.detector['pixel_size_um'] / self.lens_f_number)**2 * self.detector['optical_fill_factor']
        radiance_J_per_s_m2_nm = self.lens_transmission * self.filter_transmission * self.target_radiance 
        electrons_per_J = (1e-9 * self.wavelengths_nm / hc_Jm) * self.qe_curve

        self.Se_rate = effective_area_m2 * wavelength_step_nm * radiance_J_per_s_m2_nm * electrons_per_J

    def calculate_snr(self, exposure_time_s=1e-3, npixels=1):
        self.calculate_electron_rate()
        self.exposure_time_s = exposure_time_s
        self.Se = self.Se_rate * self.exposure_time_s
        self.Se_total = np.sum(self.Se)

        if self.detector['full_well_capacity_e'] > 0:
            self.full_well_fraction = self.Se_total / self.detector['full_well_capacity_e']
        else:
            self.full_well_fraction = 0

        dark = self.detector['dark_current_e_per_s'] * self.exposure_time_s * npixels if self.detector['dark_current_e_per_s'] > 0 else 0
        RN = self.detector['read_noise_e'] if self.detector['read_noise_e'] > 0 else 0
        signal = self.Se_total * npixels
        noise = np.sqrt(signal + dark + RN**2 * npixels)
        self.SNR = signal / noise if noise > 0 else 0
        return self.SNR

    def calculate_exposure_for_snr(self, target_snr ):
        """Calculate exposure time needed to achieve target SNR"""
        self.calculate_electron_rate()
        
        # Use iterative approach to find exposure time
        # SNR = signal / sqrt(signal + dark + RN^2)
        # where signal = Se_total * npixels * exposure_time
        
        Se_rate_total = np.sum(self.Se_rate)
        dark_rate = self.detector['dark_current_e_per_s'] if self.detector['dark_current_e_per_s'] > 0 else 0
        RN = self.detector['read_noise_e'] if self.detector['read_noise_e'] > 0 else 0
        
        # Solve quadratic equation for exposure time npixels=1
        # target_snr^2 = (Se_rate * t * npixels)^2 / (Se_rate * t * npixels + dark_rate * t * npixels + RN^2 * npixels)
        
        a = Se_rate_total**2 
        b = -target_snr**2 * (Se_rate_total + dark_rate)
        c = -target_snr**2 * RN**2 
        
        try:
            discriminant = b**2 - 4*a*c
            exposure_time = (-b + np.sqrt(discriminant)) / (2*a)
        except ValueError:
            st.error(f"❌ Unable to calculate exposure time for target SNR {target_snr}. Please check the parameters.")
            return None
        
        self.SNR = target_snr
        self.exposure_time_s = exposure_time
        self.Se = self.Se_rate * self.exposure_time_s
        self.Se_total = np.sum(self.Se)

        if self.detector['full_well_capacity_e'] > 0:
            self.full_well_fraction = self.Se_total / self.detector['full_well_capacity_e']
        else:
            self.full_well_fraction = 0

        return exposure_time

def create_plots(camera):
    """Create interactive plots using Plotly with legends """
    x = camera.wavelengths_nm
    l_min, l_max = 300, 1100
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Spectral Properties', 'Spectral Irradiance', 'Detector Signal'),
        vertical_spacing=0.1
    )
    
    # Plot 1: Spectral properties
    fig.add_trace(go.Scatter(x=x, y=camera.lens_transmission, name='Lens', 
                            line=dict(color='#2E86AB'), legendgroup='group1', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.filter_transmission, name='Filter',
                            line=dict(color='#A23B72'), legendgroup='group1', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.qe_curve, name='QE',
                            line=dict(color='#7209B7'), legendgroup='group1', showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.target_albedo, name='Albedo',
                            line=dict(color='#C73E1D'), legendgroup='group1', showlegend=True), row=1, col=1)
    
    # Plot 2: Spectral irradiance
    fig.add_trace(go.Scatter(x=x, y=camera.target_irradiance, name='Irradiance',
                            line=dict(color='#F18F01'), legendgroup='group2', showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.target_exitance, name='Exitance',
                            line=dict(color='#2F9599'), legendgroup='group2', showlegend=True), row=2, col=1)
    
    # Plot 3: Detector signal
    fig.add_trace(go.Scatter(x=x, y=camera.Se, name=f'Signal',
                            line=dict(color='#264653'), legendgroup='group3', showlegend=True), row=3, col=1)
    
    # # Update layout with legends inside plots
    # fig.update_layout(
    #     height=700,
    #     legend=dict(
    #         orientation="h",
    #         yanchor="top",
    #         y=0.98,
    #         xanchor="left",
    #         x=0.02,
    #         bgcolor="rgba(255,255,255,0.8)",
    #         bordercolor="rgba(0,0,0,0.2)",
    #         borderwidth=1
    #     )
    # )
    fig.update_layout(height=800, showlegend=True)
    
    fig.update_xaxes(title_text="Wavelength [nm]", range=[l_min, l_max], row=3, col=1)
    fig.update_yaxes(title_text="Transmission/QE", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="W/m²/nm", row=2, col=1)
    fig.update_yaxes(title_text="e⁻/nm", row=3, col=1)
    
    return fig