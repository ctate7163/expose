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

from expose import *

# Set page config
st.set_page_config(
    page_title="Camera SNR Calculator",
    page_icon="📷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    h1 {
        font-size: 2.2rem !important;
        color: #2E3B55;
        margin-bottom: 0.5rem !important;
    }
    h2 {
        font-size: 1.4rem !important;
        color: #4A5568;
        margin-bottom: 0.5rem !important;
    }
    h3 {
        font-size: 1.2rem !important;
        color: #4A5568;
        margin-bottom: 0.3rem !important;
    }
    .stDataFrame {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #CBD5E0;
        background-color: #F7FAFC;
        color: #2D3748;
    }
    .stButton > button:hover {
        background-color: #EDF2F7;
        border-color: #A0AEC0;
    }
    .stSelectbox > div > div {
        background-color: #F7FAFC;
    }
</style>
""", unsafe_allow_html=True)

# Load detector data from JSON file
@st.cache_data
def load_detectors_data():
    try:
        with open("detectors/detectors.jsonc", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("❌ detectors/detectors.jsonc file not found. Please ensure the detectors folder contains the detector specifications.")
        st.stop()

# Load CSV data with error handling
@st.cache_data
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
@st.cache_data
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


def load_preset_config(config_name, configs):
    """Load a preset configuration into session state"""
    if config_name in configs:
        config = configs[config_name]
        
        st.session_state.detector_name = config.get('detector_name', 'PYTHON-5000')
        st.session_state.band = config.get('band', 'MONO')
        
        lens = config.get('lens', {})
        st.session_state.focal_length = np.float64(lens.get('focal_length_mm', 100.0))
        st.session_state.f_number = np.float64(lens.get('f_number', 3.0))
        st.session_state.lens_transmission = np.float64(lens.get('mean_transmission', 0.9))
        
        filter_params = config.get('filter', {})
        st.session_state.cut_on_nm = filter_params.get('cut_on_nm', 400)
        st.session_state.cut_off_nm = filter_params.get('cut_off_nm', 700)
        st.session_state.filter_transmission = np.float64(filter_params.get('peak_transmission', 0.95))
        
        target = config.get('target', {})
        st.session_state.distance_AU = target.get('distance_AU', 1.0)
        albedo = target.get('albedo', [0.2, 0.2])
        st.session_state.albedo_blue = albedo[0] if isinstance(albedo, list) else albedo
        st.session_state.albedo_red = albedo[1] if isinstance(albedo, list) else albedo
        st.session_state.incidence_angle = target.get('incidence_angle_deg', 30.0)
        st.session_state.optical_depth = target.get('optical_depth', 0.0)
        st.session_state.enable_atm_correction = st.session_state.optical_depth > 0
        
        exposure = config.get('exposure', {})
        st.session_state.exposure_time_ms = exposure.get('exposure_time_ms', 1.0)
        
        st.session_state.force_calculation = True

def calculate_camera_snr(calculate_exposure=False, target_snr=100):
    """Calculate camera SNR or exposure time"""
    try:
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
        target_name = st.session_state.get('target_name', 'Target')
        
        camera = Camera(detector_name, band=band)
        camera.set_lens(focal_length, f_number, lens_transmission)
        camera.set_filter(cut_on_nm, cut_off_nm, filter_transmission)
        camera.set_target_solar_albedo_incidence(
            distance_AU=distance_AU,
            albedo=[albedo_blue, albedo_red],
            incidence_angle_deg=incidence_angle,
            optical_depth=optical_depth,
            target_name=target_name
        )
        
        if calculate_exposure:
            exposure_time_s = camera.calculate_exposure_for_snr(target_snr)
            st.session_state.calculated_exposure_ms = exposure_time_s * 1000
        else:
            snr = camera.calculate_snr(exposure_time_s=exposure_time_ms * 1e-3)
        
        st.session_state.camera = camera
        st.session_state.results_calculated = True
        
    except Exception as e:
        st.error(f"❌ Error in calculation: {str(e)}")
        st.session_state.results_calculated = False

def format_configuration_json(camera, config_name="Custom Configuration"):
    """Format configuration in the same style as cameras.json"""
    config = {
        config_name: {
            "name": config_name,
            "description": f"Custom configuration - {camera.detector_name} {camera.detector['band']} band",
            "detector_name": camera.detector_name,
            "band": camera.detector['band'],
            "lens": {
                "focal_length_mm": camera.lens_focal_length_mm,
                "f_number": camera.lens_f_number,
                "mean_transmission": camera.lens_mean_transmission
            },
            "filter": {
                "cut_on_nm": camera.filter_cut_on_nm,
                "cut_off_nm": camera.filter_cut_off_nm,
                "peak_transmission": camera.filter_transmission_peak
            },
            "target": {
                "distance_AU": camera.distance_AU,
                "albedo": [camera.target_albedo_blue, camera.target_albedo_red],
                "incidence_angle_deg": getattr(camera, 'atmospheric_correction_incidence_angle_deg', 30.0),
                "optical_depth": getattr(camera, 'atmospheric_correction_optical_depth', 0.0),
                "target_name": getattr(camera, 'target_name', '')
            },
            "exposure": {
                "exposure_time_ms": camera.exposure_time_s * 1000
            },
            "results": {
                "SNR": float(camera.SNR),
                "signal_total_e": float(camera.Se_total),
                "full_well_fraction": float(camera.full_well_fraction),
                "fov_H_deg": float(camera.fov_H_deg),
                "fov_V_deg": float(camera.fov_V_deg),
                "ifov_urad": float(camera.ifov_rad * 1e6),
                "diffraction_limit_urad": float(camera.diffraction_limit_rad * 1e6)
            }
        }
    }
    return json.dumps(config, indent=2)

def main():
    st.title("📷 Camera SNR Calculator") # 🔬📷
    st.markdown("*Radiometric performance analysis for imaging systems*")
    
    # Load data
    detectors_data = load_detectors_data()
    camera_configs = load_camera_configs()
    
    if 'show_calculator' not in st.session_state:
        st.session_state.show_calculator = False
    
    if not st.session_state.show_calculator:
        
        st.subheader("Example Configurations")
        
        if camera_configs:
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
        
        if st.button("⚙️ Manual Configuration", type="primary", use_container_width=True):
            st.session_state.show_calculator = True
            st.rerun()
        # Landing page
        st.subheader("Available Detectors")
        
        if detectors_data:
            detector_df = pd.DataFrame.from_dict(detectors_data, orient='index')
            display_cols = ['detector_type', 'manufacturer', 'pixel_count_H', 'pixel_count_V', 
                           'pixel_size_um', 'read_noise_e', 'full_well_capacity_e']
            available_cols = [col for col in display_cols if col in detector_df.columns]
            detector_df_display = detector_df[available_cols].copy()
            
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
        
    else:
        # Calculator interface
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Examples", use_container_width=True):
                st.session_state.show_calculator = False
                st.rerun()
        
        # Sidebar configuration
        st.sidebar.subheader("Configuration")
        
        detector_name = st.sidebar.selectbox(
            "Detector",
            options=list(detectors_data.keys()) if detectors_data else [],
            index=list(detectors_data.keys()).index(st.session_state.get('detector_name', list(detectors_data.keys())[0])) if detectors_data and st.session_state.get('detector_name') in detectors_data else 0,
            key='detector_name'
        )
        
        band = st.sidebar.selectbox(
            "Band",
            options=["MONO", "RED", "GREEN", "BLUE"],
            index=["MONO", "RED", "GREEN", "BLUE"].index(st.session_state.get('band', 'MONO')),
            key='band'
        )
        
        st.sidebar.markdown("**Lens**")
        focal_length = st.sidebar.number_input(
            "Focal Length (mm)", 0.0, 10000.0,
            value=st.session_state.get('focal_length', 100.0), 
            key='focal_length',
            step=1.0
        )
        f_number = st.sidebar.number_input(
            "F-number", 1.0, 20.0,
            value=st.session_state.get('f_number', 3.0), 
            key='f_number',
            step=0.1
        )
        lens_transmission = st.sidebar.slider(
            "Transmission", 0.1, 1.0, 
            st.session_state.get('lens_transmission', 0.9), 0.05,
            key='lens_transmission'
        )
        
        st.sidebar.markdown("**Filter**")
        cut_on_nm = st.sidebar.number_input(
            "Cut-on (nm)", 
            value=st.session_state.get('cut_on_nm', 400), 
            min_value=300, max_value=1000,
            key='cut_on_nm',
            step=10
        )
        cut_off_nm = st.sidebar.number_input(
            "Cut-off (nm)", 
            value=st.session_state.get('cut_off_nm', 700), 
            min_value=400, max_value=1100,
            key='cut_off_nm',
            step=10
        )
        filter_transmission = st.sidebar.slider(
            "Peak Transmission", 0.1, 1.0, 
            st.session_state.get('filter_transmission', 0.95), 0.05,
            key='filter_transmission'
        )
        
        st.sidebar.markdown("**Target**")
        distance_AU = st.sidebar.number_input(
            "Distance (AU)", 
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
            "Incidence Angle (°)", 0.0, 90.0, 
            st.session_state.get('incidence_angle', 30.0), 1.0,
            key='incidence_angle'
        )
        
        enable_atm_correction = st.sidebar.checkbox(
            "Atmospheric Correction", 
            value=st.session_state.get('enable_atm_correction', False),
            key='enable_atm_correction'
        )
        if enable_atm_correction:
            optical_depth = st.sidebar.slider(
                "Optical Depth", 0.0, 2.0, 
                st.session_state.get('optical_depth', 0.1), 0.01,
                key='optical_depth'
            )
        else:
            st.session_state.optical_depth = 0.0
        
        st.sidebar.markdown("**Exposure**")
        exposure_mode = st.sidebar.radio(
            "Calculate:",
            options=["SNR for given exposure", "Exposure for given SNR"],
            key='exposure_mode'
        )
        
        if exposure_mode == "SNR for given exposure":
            exposure_time_ms = st.sidebar.number_input(
                "Exposure Time (ms)", 
                value=st.session_state.get('exposure_time_ms', 1.0), 
                min_value=0.0001, max_value=10000.0,
                key='exposure_time_ms'
            )
            calculate_exposure = False
            target_snr = 100  # Not used in this mode
        else:
            target_snr = st.sidebar.number_input(
                "Target SNR", 
                value=st.session_state.get('target_snr', 100.0), 
                min_value=0.1, max_value=1000.0,
                key='target_snr'
            )
            calculate_exposure = True
            exposure_time_ms = 1.0  # Will be calculated
        
        # Auto-calculate when parameters change
        if ('results_calculated' not in st.session_state or 
            st.session_state.get('force_calculation', False)):
            st.session_state.force_calculation = True
            calculate_camera_snr(calculate_exposure, target_snr if calculate_exposure else 100)
        
        # Display results
        if st.session_state.get('results_calculated', False):
            camera = st.session_state.camera
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Results Summary")
                
                # Build target description
                target_desc = f"d={camera.distance_AU:.2f} AU, a=[{camera.target_albedo_blue:.2f},{camera.target_albedo_red:.2f}]"
                if hasattr(camera, 'target_name') and camera.target_name:
                    target_desc = f"{camera.target_name} at {target_desc}"
                
                results_data = {
                    'Parameter': [
                        'Detector',
                        'Lens',
                        'Resolution',
                        'FOV',
                        'Filter',
                        'Target',
                        'Exposure Time',
                        'Signal',
                        'SNR'
                    ],
                    'Values': [
                        f"{camera.detector_name} {camera.detector['band']}, pitch={camera.detector['pixel_size_um']} μm",
                        f"fl={camera.lens_focal_length_mm:.1f} mm, f/{camera.lens_f_number:.1f}, ap={camera.lens_aperture_mm:.1f} mm, T={camera.lens_mean_transmission:.2f}",
                        f"IFOV= {camera.ifov_rad*1e6:.1f} μrad, Diff= {camera.diffraction_limit_rad*1e6:.1f} μrad",
                        f"{camera.fov_H_deg:.2f}° × {camera.fov_V_deg:.2f}°",
                        f"{camera.filter_cut_on_nm}-{camera.filter_cut_off_nm} nm, T={camera.filter_transmission_peak:.2f}",
                        target_desc,
                        f"{camera.exposure_time_s*1e3:.3f} ms" + (f" (calculated)" if calculate_exposure else ""),
                        f"{camera.Se_total:.0f} e⁻, FW= {100*camera.full_well_fraction:.1f}%",
                        f"{camera.SNR:.1f}"
                    ]
                }
                
                df_results = pd.DataFrame(results_data)
                st.dataframe(
                    df_results, 
                    hide_index=True, 
                    use_container_width=True,
                    height=len(results_data['Parameter']) * 40 + 50  # Dynamic height
                )
                
                # Warnings
                if camera.full_well_fraction > 1.0:
                    st.error("⚠️ Detector saturated!")
                elif camera.full_well_fraction > 0.8:
                    st.warning("⚠️ Detector near saturation")
                
                if calculate_exposure and hasattr(st.session_state, 'calculated_exposure_ms'):
                    if st.session_state.calculated_exposure_ms > 1000:
                        st.warning(f"⚠️ Long exposure required: {st.session_state.calculated_exposure_ms/1000:.1f}s")
            
            with col2:
                st.subheader("Spectral Analysis")
                fig = create_plots(camera)
                st.plotly_chart(fig, use_container_width=True)
            
            # Export section
            st.subheader("Export")
            col1, col2, col3 = st.columns(3)
            col1, col2 = st.columns(2)
            
            with col2:
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
                    label="📊 Spectral Data (CSV)",
                    data=csv_string,
                    file_name=f"camera_analysis_{detector_name}_{band}.csv",
                    mime="text/csv"
                )
            
            with col1:
                # Configuration JSON download
                config_json = format_configuration_json(camera)
                st.download_button(
                    label="⚙️ Configuration and Results (JSON)",
                    data=config_json,
                    file_name=f"camera_config_{detector_name}_{band}.json",
                    mime="application/json"
                )
            
        #  with col3:
        #         # Results summary
        #         summary_text = f"""Configuration Summary:
        #                             Detector: {camera.detector_name} {camera.detector['band']} band
        #                             Lens: {camera.lens_focal_length_mm:.1f}mm f/{camera.lens_f_number:.1f}
        #                             Filter: {camera.filter_cut_on_nm}-{camera.filter_cut_off_nm} nm
        #                             Target: {target_desc}
        #                             Exposure: {camera.exposure_time_s*1e3:.3f} ms
        #                             SNR: {camera.SNR:.1f}
        #                             Signal: {camera.Se_total:.0f} e⁻"""
                
        #         st.download_button(
        #             label="📝 Summary (TXT)",
        #             data=summary_text,
        #             file_name=f"camera_summary_{detector_name}_{band}.txt",
        #             mime="text/plain"
        #         )   
        
        else:
            st.info("Calculating...")

if __name__ == "__main__":
    main()