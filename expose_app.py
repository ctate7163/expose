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

# Set page config
st.set_page_config(
    page_title="Camera SNR Calculator",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load detector data from JSON file
@st.cache_data
def load_detectors_data():
    try:
        with open("detectors/detectors.jsonc", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("detectors.jsonc file not found. Please ensure the detectors folder contains the detector specifications.")
        return {}

# Load CSV data with caching
@st.cache_data
def load_csv_data(filepath):
    try:
        return np.loadtxt(filepath, delimiter=',', skiprows=1)
    except FileNotFoundError:
        st.warning(f"CSV file not found: {filepath}")
        return None

# Load predefined camera configurations
@st.cache_data
def load_camera_configs():
    try:
        with open("cameras/cameras.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return default configuration if file doesn't exist
        return {
            "Mars 2020 LCAM": {
                "name": "Mars 2020 LCAM",
                "description": "Mars 2020 Left Context Camera",
                "detector_name": "PYTHON-5000",
                "band": "MONO",
                "lens": {
                    "focal_length_mm": 5.6,
                    "f_number": 2.7,
                    "mean_transmission": 0.82
                },
                "filter": {
                    "cut_on_nm": 475,
                    "cut_off_nm": 725,
                    "peak_transmission": 0.98
                },
                "target": {
                    "distance_AU": 1.57,
                    "albedo": [0.05, 0.166],
                    "incidence_angle_deg": 65.5,
                    "optical_depth": 0.0,
                    "target_name": "Mars"
                },
                "exposure": {
                    "exposure_time_ms": 0.21
                }
            }
        }

class Camera:
    def __init__(self, detector_name, band='MONO', pixel_binning=False, wavelengths_nm=None):
        if wavelengths_nm is None:
            self.wavelengths_nm = np.arange(300, 1100, 1)
        else:
            self.wavelengths_nm = wavelengths_nm

        self.detector_name = detector_name
        self.detectors = load_detectors_data()
        
        if detector_name not in self.detectors:
            raise ValueError(f"Unknown detector: {detector_name}")
        
        self.detector = self.detectors[detector_name].copy()
        self.detector['band'] = band
        
        # Load QE curve from CSV file
        qe_filename = self.detector['qe_curve_csv'][:-4] + '_' + band + '.csv'
        qe_filepath = f"detectors/{qe_filename}"
        
        qe_data = load_csv_data(qe_filepath)
        if qe_data is not None:
            wav = qe_data[:, 0]
            qe = qe_data[:, 1]
            self.qe_curve = np.interp(self.wavelengths_nm, wav, qe, left=0, right=0)
        else:
            # Fallback to mock data if CSV not found
            self.qe_curve = self._generate_mock_qe_curve(detector_name, band, self.wavelengths_nm)
        
        self.pixel_binning = pixel_binning

    def _generate_mock_qe_curve(self, detector_name, band, wavelengths):
        """Generate realistic QE curves for different detectors and bands (fallback)"""
        base_qe = np.zeros_like(wavelengths, dtype=float)
        
        if "PYTHON" in detector_name:
            peak_qe = 0.65
            peak_wavelength = 550
        elif "KAI" in detector_name or "KAF" in detector_name:
            peak_qe = 0.75
            peak_wavelength = 600
        elif "CIS" in detector_name:
            peak_qe = 0.6
            peak_wavelength = 520
        else:
            peak_qe = 0.5
            peak_wavelength = 550
        
        for i, wl in enumerate(wavelengths):
            if wl < 300 or wl > 1100:
                base_qe[i] = 0
            elif wl < 400:
                base_qe[i] = peak_qe * 0.1 * np.exp(-((wl - 400) / 50)**2)
            elif wl > 900:
                base_qe[i] = peak_qe * 0.3 * np.exp(-((wl - 900) / 100)**2)
            else:
                base_qe[i] = peak_qe * np.exp(-0.5 * ((wl - peak_wavelength) / 200)**2)
        
        if band == "RED":
            red_filter = np.where(wavelengths > 600, 1.0, np.exp(-((wavelengths - 600) / 100)**2))
            base_qe *= red_filter
        elif band == "GREEN":
            green_filter = np.exp(-((wavelengths - 530) / 80)**2)
            base_qe *= green_filter
        elif band == "BLUE":
            blue_filter = np.where(wavelengths < 550, 1.0, np.exp(-((wavelengths - 450) / 80)**2))
            base_qe *= blue_filter
        
        return np.maximum(0, base_qe)

    def set_lens(self, focal_length_mm, f_number, mean_transmission=0.9):
        self.lens_transmission = mean_transmission * np.ones_like(self.wavelengths_nm)
        self.lens_f_number = f_number
        self.lens_focal_length_mm = focal_length_mm
        self.lens_aperture_mm = focal_length_mm / f_number

        self.fov_H_deg = 2 * np.arctan((self.detector['pixel_size_um'] * 1e-6 * self.detector['pixel_count_H']) / 
                                       (2 * self.lens_focal_length_mm * 1e-3)) * 180 / np.pi
        self.fov_V_deg = 2 * np.arctan((self.detector['pixel_size_um'] * 1e-6 * self.detector['pixel_count_V']) / 
                                       (2 * self.lens_focal_length_mm * 1e-3)) * 180 / np.pi
        self.ifov_rad = (self.detector['pixel_size_um'] / 1e6) / (self.lens_focal_length_mm * 1e-3)

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
        if solar_data is not None:
            wav = solar_data[:, 0]
            rad = solar_data[:, 1] * 1e-3  # Convert from W/m2/micron to W/m2/nm
            solar_radiance_1AU = np.interp(self.wavelengths_nm, wav, rad, left=0, right=0)
        else:
            # Fallback to mock solar spectrum
            solar_radiance_1AU = self._generate_mock_solar_spectrum(self.wavelengths_nm)

        def albedoSpectrum(v_, r_, l=self.wavelengths_nm):
            l_ = [300, 450, 575, 1100]
            albedo_ = [v_, v_, r_, r_]
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

        self.target_irradiance = (1/distance_AU)**2 * solar_radiance_1AU * np.cos(np.radians(incidence_angle_deg))
        self.target_exitance = self.target_irradiance * self.target_albedo
        self.target_radiance = self.target_exitance

        if optical_depth == 0:
            self.atmospheric_correction_factor = 1.0
        else:
            self.atmospheric_correction(incidence_angle_deg, optical_depth)

    def _generate_mock_solar_spectrum(self, wavelengths):
        """Generate mock solar spectrum data (fallback)"""
        solar_irradiance = np.zeros_like(wavelengths, dtype=float)
        
        for i, wl in enumerate(wavelengths):
            if 300 <= wl <= 1100:
                solar_irradiance[i] = 1.5 * np.exp(-((wl - 500) / 300)**2) + 0.5
                if 760 <= wl <= 770:
                    solar_irradiance[i] *= 0.7
                if 940 <= wl <= 950:
                    solar_irradiance[i] *= 0.8
        
        return solar_irradiance

    def atmospheric_correction(self, incidence_angle_deg=0.0, optical_depth=0.0):
        self.atmospheric_correction_optical_depth = optical_depth
        if incidence_angle_deg < 0 or incidence_angle_deg > 90:
            raise ValueError(f"Invalid incidence angle: {incidence_angle_deg} degrees. Must be between 0 and 90 degrees.")
        self.atmospheric_correction_incidence_angle_deg = incidence_angle_deg

        self.atmospheric_correction_factor = np.exp(-optical_depth / 6 / np.cos(np.radians(incidence_angle_deg)))
        
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

        if self.detector['full_well_capacity_e'] > 0:  # Handle detectors with undefined full well
            self.full_well_fraction = self.Se_total / self.detector['full_well_capacity_e']
        else:
            self.full_well_fraction = 0

        dark = self.detector['dark_current_e_per_s'] * self.exposure_time_s * npixels if self.detector['dark_current_e_per_s'] > 0 else 0
        RN = self.detector['read_noise_e'] if self.detector['read_noise_e'] > 0 else 0
        signal = self.Se_total * npixels
        noise = np.sqrt(signal + dark + RN**2 * npixels)
        self.SNR = signal / noise if noise > 0 else 0
        return self.SNR

def create_plots(camera):
    """Create interactive plots using Plotly"""
    x = camera.wavelengths_nm
    l_min, l_max = 300, 1100
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Spectral Properties', 'Spectral Irradiance', 'Detector Signal'),
        vertical_spacing=0.08
    )
    
    # Plot 1: Spectral properties
    fig.add_trace(go.Scatter(x=x, y=camera.lens_transmission, name='Lens Transmission', 
                            line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.filter_transmission, name='Filter Transmission',
                            line=dict(color='green')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.qe_curve, name='Quantum Efficiency',
                            line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.target_albedo, name='Target Albedo',
                            line=dict(color='orange')), row=1, col=1)
    
    # Plot 2: Spectral irradiance
    fig.add_trace(go.Scatter(x=x, y=camera.target_irradiance, name='Irradiance on Target',
                            line=dict(color='purple')), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=camera.target_exitance, name='Exitance from Target',
                            line=dict(color='brown')), row=2, col=1)
    
    # Plot 3: Detector signal
    fig.add_trace(go.Scatter(x=x, y=camera.Se, name=f'Total Signal: {camera.Se_total:.0f} e⁻',
                            line=dict(color='black')), row=3, col=1)
    
    # Update layout
    fig.update_layout(height=800, showlegend=True)
    fig.update_xaxes(title_text="Wavelength [nm]", range=[l_min, l_max], row=3, col=1)
    fig.update_yaxes(title_text="Dimensionless", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="W/m²/nm", row=2, col=1)
    fig.update_yaxes(title_text="e⁻/nm", row=3, col=1)
    
    return fig

def load_preset_config(config_name, configs):
    """Load a preset configuration into session state"""
    if config_name in configs:
        config = configs[config_name]
        
        # Update session state with preset values
        st.session_state.detector_name = config.get('detector_name', 'PYTHON-5000')
        st.session_state.band = config.get('band', 'MONO')
        
        # Lens parameters
        lens = config.get('lens', {})
        st.session_state.focal_length = lens.get('focal_length_mm', 100.0)
        st.session_state.f_number = lens.get('f_number', 3.0)
        st.session_state.lens_transmission = lens.get('mean_transmission', 0.9)
        
        # Filter parameters
        filter_params = config.get('filter', {})
        st.session_state.cut_on_nm = filter_params.get('cut_on_nm', 400)
        st.session_state.cut_off_nm = filter_params.get('cut_off_nm', 700)
        st.session_state.filter_transmission = filter_params.get('peak_transmission', 0.95)
        
        # Target parameters
        target = config.get('target', {})
        st.session_state.distance_AU = target.get('distance_AU', 1.0)
        albedo = target.get('albedo', [0.2, 0.2])
        st.session_state.albedo_blue = albedo[0] if isinstance(albedo, list) else albedo
        st.session_state.albedo_red = albedo[1] if isinstance(albedo, list) else albedo
        st.session_state.incidence_angle = target.get('incidence_angle_deg', 30.0)
        st.session_state.optical_depth = target.get('optical_depth', 0.0)
        st.session_state.enable_atm_correction = st.session_state.optical_depth > 0
        
        # Exposure parameters
        exposure = config.get('exposure', {})
        st.session_state.exposure_time_ms = exposure.get('exposure_time_ms', 1.0)
        
        # Force calculation
        st.session_state.force_calculation = True

def calculate_camera_snr():
    """Calculate camera SNR with current parameters"""
    try:
        # Get parameters from session state
        detector_name = st.session_state.get('detector_name', 'PYTHON-5000')
        band = st.session_state.get('band', 'MONO')
        focal_length = st.session_state.get('focal_length', 100.0)
        f_number = st.session_state.get('f_number', 3.0)
        lens_transmission = st.session_state.get('lens_transmission', 0.9)
        cut_on_nm = st.session_state.get('cut_on_nm', 400)
        cut_off_nm = st.session_state.get('cut_off_nm', 700)
        filter_transmission = st.session_state.get('filter_transmission', 0.95)
        distance_AU = st.session_state.get('distance_AU', 1.0)
        albedo_blue = st.session_state.get('albedo_blue', 0.2)
        albedo_red = st.session_state.get('albedo_red', 0.2)
        incidence_angle = st.session_state.get('incidence_angle', 30.0)
        optical_depth = st.session_state.get('optical_depth', 0.0)
        exposure_time_ms = st.session_state.get('exposure_time_ms', 1.0)
        
        # Create camera object
        camera = Camera(detector_name, band=band)
        camera.set_lens(focal_length, f_number, lens_transmission)
        camera.set_filter(cut_on_nm, cut_off_nm, filter_transmission)
        camera.set_target_solar_albedo_incidence(
            distance_AU=distance_AU,
            albedo=[albedo_blue, albedo_red],
            incidence_angle_deg=incidence_angle,
            optical_depth=optical_depth,
            target_name="Target"
        )
        
        # Calculate SNR
        snr = camera.calculate_snr(exposure_time_s=exposure_time_ms * 1e-3)
        
        # Store results in session state
        st.session_state.camera = camera
        st.session_state.results_calculated = True
        
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")
        st.session_state.results_calculated = False

def main():
    st.title("📷 Camera SNR Calculator")
    st.markdown("Calculate Signal-to-Noise Ratio for camera systems with different detectors, optics, and targets.")
    
    # Load data
    detectors_data = load_detectors_data()
    camera_configs = load_camera_configs()
    
    # Check if we should show the landing page or the calculator
    if 'show_calculator' not in st.session_state:
        st.session_state.show_calculator = False
    
    if not st.session_state.show_calculator:
        # Landing page
        st.header("📋 Available Detectors")
        
        if detectors_data:
            detector_df = pd.DataFrame.from_dict(detectors_data, orient='index')
            display_cols = ['detector_type', 'manufacturer', 'pixel_count_H', 'pixel_count_V', 
                           'pixel_size_um', 'read_noise_e', 'full_well_capacity_e']
            available_cols = [col for col in display_cols if col in detector_df.columns]
            detector_df_display = detector_df[available_cols].copy()
            
            # Rename columns for better display
            column_names = {
                'detector_type': 'Type',
                'manufacturer': 'Manufacturer', 
                'pixel_count_H': 'H Pixels',
                'pixel_count_V': 'V Pixels',
                'pixel_size_um': 'Pixel Size (μm)',
                'read_noise_e': 'Read Noise (e⁻)',
                'full_well_capacity_e': 'Full Well (e⁻)'
            }
            detector_df_display.columns = [column_names.get(col, col) for col in detector_df_display.columns]
            st.dataframe(detector_df_display, use_container_width=True)
        
        # Preset configurations
        st.header("🎯 Example Configurations")
        st.markdown("Click on any example below to load predefined camera settings:")
        
        cols = st.columns(min(len(camera_configs), 3))
        for i, (config_name, config) in enumerate(camera_configs.items()):
            with cols[i % 3]:
                if st.button(
                    f"**{config_name}**\n\n{config.get('description', '')}", 
                    key=f"preset_{config_name}",
                    use_container_width=True
                ):
                    load_preset_config(config_name, camera_configs)
                    st.session_state.show_calculator = True
                    st.rerun()
        
        # Manual configuration button
        st.markdown("---")
        if st.button("🔧 Manual Configuration", type="primary", use_container_width=True):
            st.session_state.show_calculator = True
            st.rerun()
    
    else:
        # Calculator interface
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back to Examples", use_container_width=True):
                st.session_state.show_calculator = False
                st.rerun()
        
        # Sidebar for inputs
        st.sidebar.header("Camera Configuration")
        
        # Detector selection
        detector_name = st.sidebar.selectbox(
            "Select Detector",
            options=list(detectors_data.keys()) if detectors_data else [],
            index=list(detectors_data.keys()).index(st.session_state.get('detector_name', list(detectors_data.keys())[0])) if detectors_data and st.session_state.get('detector_name') in detectors_data else 0,
            key='detector_name'
        )
        
        band = st.sidebar.selectbox(
            "Select Band",
            options=["MONO", "RED", "GREEN", "BLUE"],
            index=["MONO", "RED", "GREEN", "BLUE"].index(st.session_state.get('band', 'MONO')),
            key='band'
        )
        
        # Lens parameters
        st.sidebar.subheader("Lens Parameters")
        focal_length = st.sidebar.number_input(
            "Focal Length (mm)", 
            value=st.session_state.get('focal_length', 100.0), 
            min_value=1.0, max_value=1000.0,
            key='focal_length'
        )
        f_number = st.sidebar.number_input(
            "F-number", 
            value=st.session_state.get('f_number', 3.0), 
            min_value=1.0, max_value=20.0,
            key='f_number'
        )
        lens_transmission = st.sidebar.slider(
            "Mean Transmission", 0.1, 1.0, 
            st.session_state.get('lens_transmission', 0.9), 0.05,
            key='lens_transmission'
        )
        
        # Filter parameters
        st.sidebar.subheader("Filter Parameters")
        cut_on_nm = st.sidebar.number_input(
            "Cut-on Wavelength (nm)", 
            value=st.session_state.get('cut_on_nm', 400), 
            min_value=300, max_value=1000,
            key='cut_on_nm'
        )
        cut_off_nm = st.sidebar.number_input(
            "Cut-off Wavelength (nm)", 
            value=st.session_state.get('cut_off_nm', 700), 
            min_value=400, max_value=1100,
            key='cut_off_nm'
        )
        filter_transmission = st.sidebar.slider(
            "Peak Transmission", 0.1, 1.0, 
            st.session_state.get('filter_transmission', 0.95), 0.05,
            key='filter_transmission'
        )
        
        # Target parameters
        st.sidebar.subheader("Target Parameters")
        distance_AU = st.sidebar.number_input(
            "Solar Distance (AU)", 
            value=st.session_state.get('distance_AU', 1.66), 
            min_value=0.1, max_value=10.0,
            key='distance_AU'
        )
        albedo_blue = st.sidebar.slider(
            "Albedo (Blue)", 0.0, 1.0, 
            st.session_state.get('albedo_blue', 0.2), 0.01,
            key='albedo_blue'
        )
        albedo_red = st.sidebar.slider(
            "Albedo (Red-NIR)", 0.0, 1.0, 
            st.session_state.get('albedo_red', 0.2), 0.01,
            key='albedo_red'
        )
        incidence_angle = st.sidebar.slider(
            "Incidence Angle (degrees)", 0.0, 90.0, 
            st.session_state.get('incidence_angle', 30.0), 1.0,
            key='incidence_angle'
        )
        
        # Atmospheric correction
        st.sidebar.subheader("Atmospheric Correction")
        enable_atm_correction = st.sidebar.checkbox(
            "Enable Atmospheric Correction", 
            value=st.session_state.get('enable_atm_correction', False),
            key='enable_atm_correction'
        )
        optical_depth = 0.0
        if enable_atm_correction:
            optical_depth = st.sidebar.slider(
                "Optical Depth", 0.0, 2.0, 
                st.session_state.get('optical_depth', 0.1), 0.01,
                key='optical_depth'
            )
        else:
            st.session_state.optical_depth = 0.0
        
        # Exposure time
        st.sidebar.subheader("Exposure Parameters")
        exposure_time_ms = st.sidebar.number_input(
            "Exposure Time (ms)", 
            value=st.session_state.get('exposure_time_ms', 1.0), 
            min_value=0.001, max_value=1000.0,
            key='exposure_time_ms'
        )
        
        # Auto-calculate when parameters change or when forced
        if ('results_calculated' not in st.session_state or 
            st.session_state.get('force_calculation', False)):
            st.session_state.force_calculation = False
            calculate_camera_snr()
        
        # Display results
        if st.session_state.get('results_calculated', False):
            camera = st.session_state.camera
            
            # Results summary
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Numerical Results")
                
                # Create results dataframe
                results_data = {
                    'Parameter': [
                        'Detector', 'Band', 'Pixel Size', 'Lens', 'FOV (H × V)', 'IFOV',
                        'Filter', 'Target Distance', 'Albedo (Blue/Red)', 'Exposure Time',
                        'Full Well Usage', 'Signal', 'SNR'
                    ],
                    'Value': [
                        f"{camera.detector_name} ({camera.detector.get('detector_type', 'N/A')})",
                        camera.detector['band'],
                        f"{camera.detector['pixel_size_um']} μm",
                        f"fl={camera.lens_focal_length_mm} mm, f/{camera.lens_f_number:.2f}",
                        f"{camera.fov_H_deg:.2f}° × {camera.fov_V_deg:.2f}°",
                        f"{camera.ifov_rad*1e6:.2f} μrad",
                        f"{camera.filter_cut_on_nm}-{camera.filter_cut_off_nm} nm",
                        f"{camera.distance_AU:.2f} AU",
                        f"{camera.target_albedo_blue:.2f} / {camera.target_albedo_red:.2f}",
                        f"{camera.exposure_time_s*1e3:.3f} ms",
                        f"{100*camera.full_well_fraction:.1f}%",
                    f"{camera.Se_total:.0f} e⁻",
                    f"{camera.SNR:.1f}"
                ]
            }
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, hide_index=True, use_container_width=True)
            
            # Warning for saturation
            if camera.full_well_fraction > 1.0:
                st.error("⚠️ Warning: Full well capacity exceeded! The detector is saturated.")
            elif camera.full_well_fraction > 0.8:
                st.warning("⚠️ Warning: Approaching full well capacity.")
        
        with col2:
            st.subheader("📈 Spectral Analysis")
            
            # Create and display interactive plot
            fig = create_plots(camera)
            st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.subheader("💾 Export Results")
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = pd.DataFrame({
                'Wavelength_nm': camera.wavelengths_nm,
                'QE': camera.qe_curve,
                'Filter_Transmission': camera.filter_transmission,
                'Lens_Transmission': camera.lens_transmission,
                'Target_Radiance': camera.target_radiance,
                'Signal_e_per_nm': camera.Se
            })
            csv_string = csv_data.to_csv(index=False)
            st.download_button(
                label="Download Spectral Data (CSV)",
                data=csv_string,
                file_name=f"camera_analysis_{detector_name}_{band}.csv",
                mime="text/csv"
            )
        
        with col2:
            # JSON download
            results_json = {
                'detector': camera.detector,
                'lens': {
                    'focal_length_mm': camera.lens_focal_length_mm,
                    'f_number': camera.lens_f_number,
                    'transmission': float(camera.lens_transmission[0])
                },
                'filter': {
                    'cut_on_nm': camera.filter_cut_on_nm,
                    'cut_off_nm': camera.filter_cut_off_nm,
                    'peak_transmission': camera.filter_transmission_peak
                },
                'results': {
                    'SNR': float(camera.SNR),
                    'signal_total_e': float(camera.Se_total),
                    'exposure_time_s': float(camera.exposure_time_s),
                    'full_well_fraction': float(camera.full_well_fraction)
                }
            }
            json_string = json.dumps(results_json, indent=2)
            st.download_button(
                label="Download Configuration (JSON)",
                data=json_string,
                file_name=f"camera_config_{detector_name}_{band}.json",
                mime="application/json"
            )
    
    else:
        st.info("👈 Configure your camera parameters in the sidebar and click 'Calculate SNR' to see results.")
        
        # Show detector specifications table
        st.subheader("📋 Available Detectors")
        detector_df = pd.DataFrame.from_dict(DETECTORS_DATA, orient='index')
        detector_df = detector_df[['detector_type', 'manufacturer', 'pixel_count_H', 'pixel_count_V', 
                                 'pixel_size_um', 'read_noise_e', 'full_well_capacity_e']]
        detector_df.columns = ['Type', 'Manufacturer', 'H Pixels', 'V Pixels', 'Pixel Size (μm)', 
                              'Read Noise (e⁻)', 'Full Well (e⁻)']
        st.dataframe(detector_df, use_container_width=True)

if __name__ == "__main__":
    main()
