# main_app.py
import sys
import os
import time
import json
import shutil
import traceback
import copy
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('QtAgg') # Use Qt Agg backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QComboBox, QTextEdit, QTabWidget, QGroupBox,
    QCheckBox, QLineEdit, QDoubleSpinBox, QSizePolicy, QMessageBox, QFileDialog,
    QScrollArea, QGraphicsScene, QGraphicsView, QGraphicsTextItem
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QPointF, QRectF
from PyQt6.QtGui import QFontMetrics, QIcon, QPixmap, QPainter, QColor, QFont

# Import local modules
from settings_manager import SettingsManager, DEFAULT_SETTINGS
from ekf_imu import EKF_IMU, quaternion_to_rotation_matrix, quaternion_multiply, quaternion_conjugate
from madgwick_ahrs import MadgwickAHRS
from serial_worker import SerialWorker


def ellipsoid_residuals(params, data):
    cx, cy, cz = params[0:3]
    M = np.array([
        [params[3], params[4], params[5]],
        [params[4], params[6], params[7]],
        [params[5], params[7], params[8]]
    ], dtype=float)
    center = np.array([cx, cy, cz], dtype=float)
    epsilon_reg = 1e-9
    M_reg = M + np.identity(3) * epsilon_reg
    residuals = np.zeros(len(data))
    for i, point in enumerate(data):
        diff = point - center
        val = diff.T @ M_reg @ diff
        residuals[i] = val - 1.0
    return residuals

def fit_ellipsoid(mag_data_np):
    if mag_data_np.shape[0] < 20:
        print("Not enough mag data for fitting.")
        return None, None
    mag_data_np = np.asarray(mag_data_np, dtype=float)
    if not np.all(np.isfinite(mag_data_np)):
         print("Non-finite values in mag data for fitting.")
         return None, None

    initial_center = np.mean(mag_data_np, axis=0)
    diffs = mag_data_np - initial_center
    avg_radius_sq = np.mean(np.sum(diffs**2, axis=1))
    if avg_radius_sq < 1e-6 or not np.isfinite(avg_radius_sq): avg_radius_sq = 1.0
    inv_avg_radius_sq = 1.0 / avg_radius_sq
    initial_params = np.array([
        initial_center[0], initial_center[1], initial_center[2],
        inv_avg_radius_sq, 0, 0, inv_avg_radius_sq, 0, inv_avg_radius_sq
    ], dtype=float)
    if not np.all(np.isfinite(initial_params)):
        initial_params = np.array([0,0,0, 1,0,0, 1,0, 1], dtype=float)

    try:
        result = least_squares(ellipsoid_residuals, initial_params, args=(mag_data_np,),
                               method='lm', verbose=0, ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=3000)
        if not result.success or result.status <= 0:
            print(f"Ellipsoid fitting failed. Status: {result.status}, Msg: {result.message}")
            min_vals = np.min(mag_data_np, axis=0); max_vals = np.max(mag_data_np, axis=0)
            hard_iron_offset = (max_vals + min_vals) / 2.0
            soft_iron_matrix = np.identity(3, dtype=float)
            return hard_iron_offset, soft_iron_matrix

        fitted_params = result.x
        hard_iron_offset = fitted_params[0:3]
        M = np.array([
            [fitted_params[3], fitted_params[4], fitted_params[5]],
            [fitted_params[4], fitted_params[6], fitted_params[7]],
            [fitted_params[5], fitted_params[7], fitted_params[8]]
        ], dtype=float)
        try:
            eigenvalues = np.linalg.eigvalsh(M)
            if np.any(eigenvalues <= 1e-9) or not np.all(np.isfinite(eigenvalues)):
                soft_iron_matrix = np.identity(3, dtype=float)
            else:
                M_reg = M + np.identity(3) * 1e-9
                L = np.linalg.cholesky(M_reg)
                soft_iron_matrix = L.T
                if not np.all(np.isfinite(soft_iron_matrix)):
                    soft_iron_matrix = np.identity(3, dtype=float)
        except np.linalg.LinAlgError:
            soft_iron_matrix = np.identity(3, dtype=float)

        if not np.all(np.isfinite(hard_iron_offset)): hard_iron_offset = np.zeros(3)
        if not np.all(np.isfinite(soft_iron_matrix)): soft_iron_matrix = np.identity(3)
        return hard_iron_offset, soft_iron_matrix
    except Exception as e:
        print(f"Error during ellipsoid fitting: {e}")
        traceback.print_exc()
        try:
            min_vals = np.min(mag_data_np, axis=0); max_vals = np.max(mag_data_np, axis=0)
            hard_iron_offset = (max_vals + min_vals) / 2.0
            if not np.all(np.isfinite(hard_iron_offset)): hard_iron_offset = np.zeros(3)
        except: hard_iron_offset = np.zeros(3)
        soft_iron_matrix = np.identity(3, dtype=float)
        return hard_iron_offset, soft_iron_matrix


DARK_STYLESHEET = """
    QMainWindow, QWidget {
        background-color: #2b2b2b;
        color: #e0e0e0;
        font-size: 10pt;
    }
    QPushButton {
        background-color: #3c3f41;
        border: 1px solid #555555;
        padding: 6px 10px;
        min-height: 22px;
    }
    QPushButton:hover {
        background-color: #4f5254;
    }
    QPushButton:pressed {
        background-color: #2a2d2f;
    }
    QPushButton:disabled {
        background-color: #333333;
        color: #777777;
    }
    QComboBox {
        background-color: #3c3f41;
        border: 1px solid #555555;
        padding: 4px;
        min-height: 22px;
    }
    QComboBox QAbstractItemView {
        background-color: #3c3f41;
        selection-background-color: #555a5f;
        color: #e0e0e0;
        border: 1px solid #555555;
    }
    QLineEdit, QTextEdit, QSpinBox, QDoubleSpinBox {
        background-color: #3c3f41;
        border: 1px solid #555555;
        padding: 4px;
        min-height: 22px;
        color: #e0e0e0;
    }
    QTextEdit {
        background-color: #202020;
        color: #d0d0d0;
        font-family: Consolas, monaco, monospace;
    }
    QTabWidget::pane {
        border: 1px solid #444444;
        border-top: none;
    }
    QTabBar::tab {
        background: #3c3f41;
        border: 1px solid #444444;
        border-bottom: none;
        padding: 8px 15px;
        margin-right: 1px;
    }
    QTabBar::tab:selected {
        background: #4f5254;
        border-bottom: 1px solid #4f5254;
    }
    QTabBar::tab:!selected:hover {
        background: #45484a;
    }
    QLabel {
        color: #e0e0e0;
    }
    QCheckBox {
        spacing: 5px;
        color: #e0e0e0;
    }
    QCheckBox::indicator {
        width: 13px;
        height: 13px;
    }
    QGroupBox {
        border: 1px solid #555555;
        margin-top: 15px;
        padding-top: 10px;
        font-weight: bold;
        color: #e0e0e0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top center;
        padding: 0 5px;
        background-color: #2b2b2b;
        color: #e0e0e0;
    }
    QStatusBar {
        background-color: #3c3f41;
        color: #e0e0e0;
    }
    QStatusBar::item {
        border: none;
    }
    QScrollArea {
        background-color: #2b2b2b;
        border: none;
    }
    QGraphicsView {
        background-color: transparent;
        border-style: none;
    }
"""

PLOT_COLORS = {
    'x': '#1f77b4',
    'y': '#ff7f0e',
    'z': '#2ca02c',
    'raw_mag': '#d62728',
    'calib_mag': '#9467bd'
}


class IMUCalibratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings_manager = SettingsManager()
        self.setWindowTitle(f"Versys Research IMU Calibrator v{self.settings_manager.get('app_version')}")
        icon_path = "logo.png"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))

        self.setGeometry(50, 50, 1600, 980)
        self.setStyleSheet(DARK_STYLESHEET)
        plt.style.use('dark_background')
        plt.rcParams['axes.facecolor'] = '#252525'
        plt.rcParams['figure.facecolor'] = '#2b2b2b'

        self._initialize_state_vars()
        self._setup_gui() 

        self.ahrs_algorithm = None 
        self._initialize_ahrs_algorithm() 

        self.serial_worker = SerialWorker(self.settings_manager)

        self.serial_worker.list_serial_ports()
        self.load_calibration_file() 

        self._reset_ahrs_with_current_params()

        self.gui_update_timer = QTimer(self)
        self.gui_update_timer.timeout.connect(self.update_gui_elements)
        self.gui_update_timer.start(self.settings_manager.get("gui_update_interval_ms"))

        self.plot_update_timer = QTimer(self)
        self.plot_update_timer.timeout.connect(self._update_plots)
        self.plot_update_timer.start(self.settings_manager.get("plot_update_interval_ms"))

        self.last_data_timestamp_ms = None
        self.processing_start_time = None
        self.processing_delay_ms = 0.0
        self.data_buffer = {'accel': None, 'gyro': None, 'mag': None, 'timestamp_ms': None}

        self.log_message("Application initialized.") 

        self._connect_signals()
        self.populate_settings_gui()
        self._update_button_states()


    def _initialize_ahrs_algorithm(self):
        algo_choice = self.settings_manager.get("ahrs_algorithm")
        if algo_choice == "EKF":
            self.ahrs_algorithm = EKF_IMU(self.settings_manager)
            self.log_message("Using EKF for AHRS.")
        elif algo_choice == "Madgwick":
            beta = self.settings_manager.get("madgwick_beta")
            nominal_dt = 1.0 / 100.0
            self.ahrs_algorithm = MadgwickAHRS(self.settings_manager, sample_period=nominal_dt, beta=beta)
            self.log_message(f"Using Madgwick AHRS with beta: {beta}.")
        else:
            self.log_message(f"Warning: Unknown AHRS algorithm '{algo_choice}'. Defaulting to EKF.")
            self.settings_manager.set("ahrs_algorithm", "EKF")
            self.ahrs_algorithm = EKF_IMU(self.settings_manager)

        if hasattr(self, 'ahrs_status_label'): # GUI element might not exist on first call
            self.ahrs_status_label.setText(f"Current AHRS: {self.settings_manager.get('ahrs_algorithm')}")

        self._reset_ahrs_with_current_params()


    def _reset_ahrs_with_current_params(self):
        if self.ahrs_algorithm:
            if isinstance(self.ahrs_algorithm, EKF_IMU):
                self.ahrs_algorithm.reset_state(bias_init=self.calibration_params["gyro_offset"])
            elif isinstance(self.ahrs_algorithm, MadgwickAHRS):
                self.ahrs_algorithm.reset_state() 
            self.log_message(f"AHRS ({self.settings_manager.get('ahrs_algorithm')}) reset with current calibration.")

    def _initialize_state_vars(self):
        self.calibration_params = { 
            "accel_offset": np.zeros(3, dtype=float),
            "gyro_offset": np.zeros(3, dtype=float),
            "mag_offset": np.zeros(3, dtype=float),
            "mag_matrix": np.identity(3, dtype=float),
        }
        self.accel_gyro_data_collected_flag = False
        self.mag_data_collected_flag = False
        self.calibration_is_processed_and_applied = False

        self.calibration_state = "idle" # "idle", "collecting_gyro_accel", "collecting_mag", "collecting_for_ekf_tune", "collecting_for_madgwick_beta_tune", "benchmarking_static"
        self.calib_buffers = {'accel': [], 'gyro': [], 'mag': []} 
        self.ekf_tune_buffer = {'accel': [], 'mag': []} 
        self.madgwick_tune_buffer = {'gyro': []} # For Madgwick beta tuning
        self.mag_calib_collection_buffer_np = np.empty((0,3)) 

        self.display_data_timeseries = {
            'accel': {'t': [], 'x': [], 'y': [], 'z': []},
            'gyro': {'t': [], 'x': [], 'y': [], 'z': []},
        }
        self.final_calibrated_mag_display_data = {'x': [], 'y': [], 'z': []}
        self.mag_plot_needs_final_update = False
        self.mag_plot_autoscale = True
        self.mag_plot_scaled_once = False

        self.orientation_quat_display = np.array([1.0, 0.0, 0.0, 0.0])
        self.reference_orientation_quat_display = np.array([1.0, 0.0, 0.0, 0.0])

        self.benchmark_buffer = {'accel': [], 'gyro': [], 'start_time': None}
        self.benchmark_history = []

    def _setup_gui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self.controls_panel = QWidget()
        self.controls_panel.setFixedWidth(450)
        self.controls_layout = QVBoxLayout(self.controls_panel)
        self.controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.tabs_panel = QTabWidget()

        self.main_layout.addWidget(self.controls_panel)
        self.main_layout.addWidget(self.tabs_panel)

        self._setup_controls_panel() 
        self._setup_calibration_tab()
        self._setup_benchmark_tab()
        self._setup_settings_tab()
        self._setup_about_tab()

        self.status_bar = self.statusBar()
        self.status_label_connection = QLabel("Status: Disconnected")
        self.status_bar.addWidget(self.status_label_connection)
        self.status_label_calibration = QLabel("Calibration: Idle")
        self.status_bar.addPermanentWidget(self.status_label_calibration)

    def _setup_controls_panel(self):
        # Serial Connection Group
        conn_group = QGroupBox("Serial Connection")
        conn_layout = QGridLayout()
        conn_layout.addWidget(QLabel("COM Port:"), 0, 0)
        self.port_combobox = QComboBox()
        conn_layout.addWidget(self.port_combobox, 0, 1)
        self.refresh_ports_button = QPushButton("Refresh Ports")
        conn_layout.addWidget(self.refresh_ports_button, 0, 2)
        self.connect_button = QPushButton("Connect")
        conn_layout.addWidget(self.connect_button, 1, 0, 1, 3)
        conn_group.setLayout(conn_layout)
        self.controls_layout.addWidget(conn_group)

        # Sensor Calibration Group
        cal_group = QGroupBox("Sensor Calibration")
        cal_layout = QVBoxLayout()
        self.collect_accel_gyro_button = QPushButton("1. Collect Accel/Gyro Data (Static)")
        cal_layout.addWidget(self.collect_accel_gyro_button)
        self.collect_mag_button = QPushButton("2. Collect Magnetometer Data (Moving)")
        cal_layout.addWidget(self.collect_mag_button)
        self.process_calib_button = QPushButton("3. Process Collected Data & Apply")
        cal_layout.addWidget(self.process_calib_button)
        self.save_calib_as_button = QPushButton("Save Applied Calibration As...")
        cal_layout.addWidget(self.save_calib_as_button)
        self.load_calib_button = QPushButton("Load Calibration File...")
        cal_layout.addWidget(self.load_calib_button)
        self.clear_calib_buffers_button = QPushButton("Clear Calibration Data Buffers")
        cal_layout.addWidget(self.clear_calib_buffers_button)
        self.reset_applied_calib_button = QPushButton("Reset Applied Calibration to Zero")
        cal_layout.addWidget(self.reset_applied_calib_button)
        cal_group.setLayout(cal_layout)
        self.controls_layout.addWidget(cal_group)

        # AHRS & Visualization Group
        ahrs_vis_group = QGroupBox("AHRS & Visualization")
        ahrs_vis_layout = QVBoxLayout()
        self.ahrs_status_label = QLabel(f"Current AHRS: {self.settings_manager.get('ahrs_algorithm')}")
        self.ahrs_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ahrs_vis_layout.addWidget(self.ahrs_status_label)

        self.tune_ekf_r_button = QPushButton("Tune EKF R-Values (Static)")
        ahrs_vis_layout.addWidget(self.tune_ekf_r_button)
        self.tune_madgwick_beta_button = QPushButton("Tune Madgwick Beta (Static)") # New Button
        ahrs_vis_layout.addWidget(self.tune_madgwick_beta_button)

        self.reset_ahrs_button = QPushButton("Reset AHRS State")
        ahrs_vis_layout.addWidget(self.reset_ahrs_button)

        self.set_vis_offset_button = QPushButton("Set Current Orientation as Zero")
        ahrs_vis_layout.addWidget(self.set_vis_offset_button)
        self.reset_vis_offset_button = QPushButton("Reset Orientation Zero")
        ahrs_vis_layout.addWidget(self.reset_vis_offset_button)

        mag_plot_controls_layout = QHBoxLayout()
        self.autoscale_mag_checkbox = QCheckBox("Autoscale Mag Plot")
        self.autoscale_mag_checkbox.setChecked(True)
        mag_plot_controls_layout.addWidget(self.autoscale_mag_checkbox)
        self.zoom_fit_mag_button = QPushButton("Zoom Fit Mag")
        mag_plot_controls_layout.addWidget(self.zoom_fit_mag_button)
        ahrs_vis_layout.addLayout(mag_plot_controls_layout)

        self.vis_help_button = QPushButton("Help: 3D Visualizer")
        ahrs_vis_layout.addWidget(self.vis_help_button)
        ahrs_vis_group.setLayout(ahrs_vis_layout)
        self.controls_layout.addWidget(ahrs_vis_group)

        # Log Group
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        self.log_area = QTextEdit() 
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(130) # Adjusted height
        log_layout.addWidget(self.log_area)
        log_group.setLayout(log_layout)
        self.controls_layout.addWidget(log_group)
        self.controls_layout.addStretch(1)


    def _setup_calibration_tab(self): # Unchanged
        self.calib_tab = QWidget()
        calib_tab_layout = QVBoxLayout(self.calib_tab)

        plot_widget_container = QWidget()
        plot_widget_layout = QGridLayout(plot_widget_container)
        plot_widget_layout.setContentsMargins(0,0,0,0)

        self.fig_calib = Figure(figsize=(9,9))
        self.canvas_calib = FigureCanvas(self.fig_calib)
        plot_widget_layout.addWidget(self.canvas_calib, 0, 0)

        self.overlay_scene = QGraphicsScene()
        self.overlay_view = QGraphicsView(self.overlay_scene)
        self.overlay_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.overlay_view.setStyleSheet("background: transparent; border: none;")
        self.overlay_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overlay_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overlay_view.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True) 

        self.delay_text_item = QGraphicsTextItem()
        self.delay_text_item.setDefaultTextColor(QColor("lightgray"))
        font = QFont("Consolas", 9)
        self.delay_text_item.setFont(font)
        self.overlay_scene.addItem(self.delay_text_item)

        plot_widget_layout.addWidget(self.overlay_view, 0, 0)

        gs = self.fig_calib.add_gridspec(3, 2, height_ratios=[1, 1, 1.2])
        self.ax_accel = self.fig_calib.add_subplot(gs[0, 0])
        self.ax_gyro = self.fig_calib.add_subplot(gs[1, 0], sharex=self.ax_accel)
        self.ax_mag_scatter = self.fig_calib.add_subplot(gs[2, 0], projection='3d')
        self.ax_orientation = self.fig_calib.add_subplot(gs[:, 1], projection='3d')

        self._init_plots()
        calib_tab_layout.addWidget(plot_widget_container)
        self.tabs_panel.addTab(self.calib_tab, "Calibration & Visualization")

    def _init_plots(self): # Unchanged
        self.ax_accel.set_title("Accelerometer", fontsize=10)
        self.ax_accel.set_ylabel("Accel (Units)", fontsize=9)
        self.accel_lines = {
            'x': self.ax_accel.plot([], [], label='X', lw=1.2, color=PLOT_COLORS['x'])[0],
            'y': self.ax_accel.plot([], [], label='Y', lw=1.2, color=PLOT_COLORS['y'])[0],
            'z': self.ax_accel.plot([], [], label='Z', lw=1.2, color=PLOT_COLORS['z'])[0],
        }
        self.ax_accel.legend(fontsize='x-small', loc='upper right')
        self.ax_accel.grid(True, linestyle=':', alpha=0.5)
        plt.setp(self.ax_accel.get_xticklabels(), visible=False)

        self.ax_gyro.set_title("Gyroscope", fontsize=10)
        self.ax_gyro.set_ylabel("Gyro (rad/s)", fontsize=9)
        self.ax_gyro.set_xlabel("Time (s)", fontsize=9)
        self.gyro_lines = {
            'x': self.ax_gyro.plot([], [], label='X', lw=1.2, color=PLOT_COLORS['x'])[0],
            'y': self.ax_gyro.plot([], [], label='Y', lw=1.2, color=PLOT_COLORS['y'])[0],
            'z': self.ax_gyro.plot([], [], label='Z', lw=1.2, color=PLOT_COLORS['z'])[0],
        }
        self.ax_gyro.legend(fontsize='x-small', loc='upper right')
        self.ax_gyro.grid(True, linestyle=':', alpha=0.5)

        self.ax_mag_scatter.set_title("Magnetometer", fontsize=10)
        self.mag_scatter_raw_plot = self.ax_mag_scatter.scatter([], [], [], label='Raw/Collect', s=5, alpha=0.4, c=PLOT_COLORS['raw_mag'], depthshade=True)
        self.mag_scatter_calib_plot = self.ax_mag_scatter.scatter([], [], [], label='Calibrated', s=5, alpha=0.6, c=PLOT_COLORS['calib_mag'], depthshade=True)
        self.ax_mag_scatter.set_xlabel("X", labelpad=-2, fontsize=9); self.ax_mag_scatter.set_ylabel("Y", labelpad=-2, fontsize=9); self.ax_mag_scatter.set_zlabel("Z", labelpad=-2, fontsize=9)
        self.ax_mag_scatter.tick_params(axis='both', which='major', labelsize=7, pad=-4)
        self.ax_mag_scatter.legend(fontsize='x-small', loc='upper left')
        self._set_axes_equal_3d(self.ax_mag_scatter)

        self.ax_orientation.title.set_visible(False) 
        lim = 0.8
        self.ax_orientation.set_xlim(-lim, lim); self.ax_orientation.set_ylim(-lim, lim); self.ax_orientation.set_zlim(-lim, lim)
        self.ax_orientation.set_xlabel("X (World)", fontsize=9, labelpad=0); self.ax_orientation.set_ylabel("Y (World)", fontsize=9, labelpad=0); self.ax_orientation.set_zlabel("Z (World)", fontsize=9, labelpad=0)
        self.ax_orientation.tick_params(axis='both', labelsize=7, pad=-2)

        self.cube_v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]) * 0.35
        self.cube_edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        self.cube_lines_plot = [self.ax_orientation.plot([], [], [], c=PLOT_COLORS['x'], lw=1.5, alpha=0.9)[0] for _ in self.cube_edges]
        self.coord_lines_plot = [
            self.ax_orientation.plot([], [], [], c=PLOT_COLORS['x'], lw=2.0, alpha=0.8)[0],
            self.ax_orientation.plot([], [], [], c=PLOT_COLORS['y'], lw=2.0, alpha=0.8)[0],
            self.ax_orientation.plot([], [], [], c=PLOT_COLORS['z'], lw=2.0, alpha=0.8)[0]
        ]
        self.fig_calib.tight_layout(pad=2.5)

    def _setup_benchmark_tab(self): # Unchanged
        self.benchmark_tab = QWidget()
        benchmark_layout = QVBoxLayout(self.benchmark_tab)

        controls_layout = QHBoxLayout()
        self.start_benchmark_button = QPushButton("Start Static Benchmark")
        controls_layout.addWidget(self.start_benchmark_button)
        self.clear_benchmark_history_button = QPushButton("Clear History")
        controls_layout.addWidget(self.clear_benchmark_history_button)
        benchmark_layout.addLayout(controls_layout)

        self.benchmark_results_area = QTextEdit()
        self.benchmark_results_area.setReadOnly(True)
        self.benchmark_results_area.setFontFamily("Consolas")
        self.benchmark_results_area.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.benchmark_results_area.setPlaceholderText("Benchmark results will appear here.\nHistory of results will accumulate below.")
        benchmark_layout.addWidget(self.benchmark_results_area)
        self.tabs_panel.addTab(self.benchmark_tab, "Benchmark")

    def _setup_settings_tab(self): 
        self.settings_tab_scroller = QScrollArea()
        self.settings_tab_scroller.setWidgetResizable(True)
        self.settings_tab_widget = QWidget()
        settings_tab_layout = QVBoxLayout(self.settings_tab_widget)
        self.setting_entries = {}

        general_group = QGroupBox("General Settings")
        general_layout = QGridLayout()
        row = 0
        general_settings_keys = ["default_baudrate", "gravity",
                                 "min_valid_dt_sec", "max_valid_dt_sec",
                                 "mag_calib_samples", "accel_gyro_calib_samples",
                                 "benchmark_duration_sec", "max_benchmark_history",
                                 "gui_update_interval_ms", "plot_update_interval_ms"]
        for key in general_settings_keys:
            general_layout.addWidget(QLabel(f"{key.replace('_',' ').capitalize()}:"), row, 0)
            entry = QLineEdit(str(self.settings_manager.get(key)))
            self.setting_entries[key] = entry
            general_layout.addWidget(entry, row, 1)
            row += 1
        general_group.setLayout(general_layout)
        settings_tab_layout.addWidget(general_group)

        ahrs_group = QGroupBox("AHRS Algorithm Settings")
        ahrs_layout = QGridLayout()
        row = 0
        ahrs_layout.addWidget(QLabel("Algorithm:"), row, 0)
        self.ahrs_algo_combobox = QComboBox()
        self.ahrs_algo_combobox.addItems(["EKF", "Madgwick"])
        current_algo = self.settings_manager.get("ahrs_algorithm")
        if current_algo not in ["EKF", "Madgwick"]:
            current_algo = "EKF"
            self.settings_manager.set("ahrs_algorithm", current_algo) 
        self.ahrs_algo_combobox.setCurrentText(current_algo)
        self.setting_entries["ahrs_algorithm"] = self.ahrs_algo_combobox
        ahrs_layout.addWidget(self.ahrs_algo_combobox, row, 1)
        row += 1

        # Madgwick specific widgets
        self.madgwick_beta_label = QLabel("Madgwick Beta (Gain):")
        ahrs_layout.addWidget(self.madgwick_beta_label, row, 0)
        self.madgwick_beta_entry = QLineEdit(str(self.settings_manager.get("madgwick_beta")))
        self.setting_entries["madgwick_beta"] = self.madgwick_beta_entry
        ahrs_layout.addWidget(self.madgwick_beta_entry, row, 1)
        row +=1
        
        self.madgwick_tune_widgets = []
        madgwick_tune_keys = ["madgwick_tune_samples", "min_madgwick_beta", "max_madgwick_beta"]
        for key in madgwick_tune_keys:
            label_text = f"{key.replace('madgwick_','').replace('_',' ').capitalize()} (Madgwick Tune):"
            label = QLabel(label_text)
            ahrs_layout.addWidget(label, row, 0)
            self.madgwick_tune_widgets.append(label)
            entry = QLineEdit(str(self.settings_manager.get(key)))
            self.setting_entries[key] = entry
            ahrs_layout.addWidget(entry, row, 1)
            self.madgwick_tune_widgets.append(entry)
            row +=1


        # EKF specific widgets
        self.ekf_specific_widgets = []
        ekf_settings_keys = ["ekf_q_gyro_noise", "ekf_q_gyro_bias_noise",
                           "ekf_r_accel_noise", "ekf_r_mag_noise",
                           "ekf_initial_gyro_bias_cov", "ekf_initial_orientation_cov",
                           "ekf_tune_samples", "min_ekf_r_accel_noise", "min_ekf_r_mag_noise"]
        for key in ekf_settings_keys:
            label_text = f"{key.replace('ekf_','').replace('_',' ').capitalize()}"
            if "tune" in key or "min_ekf" in key :
                label_text += " (EKF Tune)"
            else:
                label_text += " (EKF)"
            label = QLabel(label_text + ":")

            ahrs_layout.addWidget(label, row, 0)
            self.ekf_specific_widgets.append(label)
            current_val = self.settings_manager.get(key)
            entry = QLineEdit(json.dumps(current_val) if isinstance(current_val, list) else str(current_val))
            self.setting_entries[key] = entry
            ahrs_layout.addWidget(entry, row, 1)
            self.ekf_specific_widgets.append(entry)
            row += 1

        ahrs_layout.addWidget(QLabel("Mag Ref Vector NED (uT):"), row, 0)
        mag_ref_entry = QLineEdit(json.dumps(self.settings_manager.get("mag_ref_vector_ned")))
        self.setting_entries["mag_ref_vector_ned"] = mag_ref_entry
        ahrs_layout.addWidget(mag_ref_entry, row, 1)

        ahrs_group.setLayout(ahrs_layout)
        settings_tab_layout.addWidget(ahrs_group)
        self.ahrs_algo_combobox.currentTextChanged.connect(self._update_ahrs_settings_visibility)
        self._update_ahrs_settings_visibility(self.ahrs_algo_combobox.currentText())

        settings_button_layout = QHBoxLayout()
        self.save_settings_button = QPushButton("Save Settings")
        settings_button_layout.addWidget(self.save_settings_button)
        self.reset_settings_button = QPushButton("Reset to Defaults")
        settings_button_layout.addWidget(self.reset_settings_button)
        settings_tab_layout.addLayout(settings_button_layout)

        settings_tab_layout.addStretch(1)
        self.settings_tab_widget.setLayout(settings_tab_layout)
        self.settings_tab_scroller.setWidget(self.settings_tab_widget)
        self.tabs_panel.addTab(self.settings_tab_scroller, "Settings")


    def _update_ahrs_settings_visibility(self, algorithm_name):
        is_ekf = (algorithm_name == "EKF")
        is_madgwick = (algorithm_name == "Madgwick")

        # EKF specific settings and tune button
        for widget in self.ekf_specific_widgets:
            widget.setVisible(is_ekf)
        if hasattr(self, 'tune_ekf_r_button'): 
            self.tune_ekf_r_button.setVisible(is_ekf)

        # Madgwick specific settings and tune button
        self.madgwick_beta_label.setVisible(is_madgwick)
        self.madgwick_beta_entry.setVisible(is_madgwick)
        for widget in self.madgwick_tune_widgets:
            widget.setVisible(is_madgwick)
        if hasattr(self, 'tune_madgwick_beta_button'):
            self.tune_madgwick_beta_button.setVisible(is_madgwick)


    def _setup_about_tab(self): 
        self.about_tab = QWidget()
        about_layout = QVBoxLayout(self.about_tab)
        about_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        logo_label = QLabel()
        pixmap = QPixmap("VersysResearch_W.png")
        if not pixmap.isNull():
            logo_label.setPixmap(pixmap.scaledToWidth(150, Qt.TransformationMode.SmoothTransformation))
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            about_layout.addWidget(logo_label)
        title_label = QLabel(f"IMU Calibration Tool v{self.settings_manager.get('app_version')}")
        title_font = title_label.font(); title_font.setPointSize(14); title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        about_layout.addWidget(title_label)
        about_text_edit = QTextEdit()
        about_text_edit.setReadOnly(True)
        about_text_edit.setStyleSheet("background-color: #2e2e2e; border: none; font-size: 9pt;")
        about_html = f"""
        <p>
            <b>IMU Calibration tool provided by Versys Research Team</b>
            <br>
            Developer: Zeaky Nguyen (ZenD)
            <br>
            Branch: VisionSDK/Tools
            <br>
            Liscence: Apache 2.0
        </p>
        <hr>
        <h4>Technologies and Algorithms</h4>
        <p>This tool utilizes several techniques for IMU calibration and orientation estimation:</p>
        <ul>
            <li><b>Accelerometer & Gyroscope Calibration:</b> Static data collection to estimate sensor offsets (biases). Accelerometer calibration attempts to isolate gravity.</li>
            <li><b>Magnetometer Calibration:</b> Ellipsoid fitting (Least Squares) to raw magnetometer data to determine hard-iron offsets and soft-iron distortions (scale factors and cross-axis sensitivities).</li>
            <li><b>Orientation Estimation:</b> Selectable AHRS algorithms (EKF or Madgwick) fuse data from the accelerometer, gyroscope, and magnetometer to provide a robust 3D orientation estimate (quaternion-based). The EKF also estimates gyroscope biases online.</li>
            <li><b>EKF R-Value Tuning:</b> A utility to estimate accelerometer and magnetometer measurement noise variances (R values for EKF) from static data.</li><li><b>Madgwick Beta Tuning:</b> A utility to suggest a `beta` gain for the Madgwick filter based on gyroscope noise characteristics from static data.</li>
            <li><b>Data Visualization:</b> Real-time plotting of sensor data and 3D orientation.</li></ul><hr><h4>Open Source Libraries & Acknowledgements</h4><p>This application is built using several open-source libraries:</p><ul><li><b>Python:</b> The core programming language.</li>
            <li><b>PyQt6:</b> For the graphical user interface.</li><li><b>NumPy:</b> For numerical operations.</li><li><b>SciPy:</b> For optimization (ellipsoid fitting) and scientific computing.</li><li><b>Matplotlib:</b> For plotting.</li><li><b>Pyserial:</b> For serial communication.</li>
        </ul>
        <p>Special thanks to the developers and communities behind these projects.</p>
        <hr>
        <h4>Disclaimers</h4>
        <p>
            <b>NO WARRANTIES:</b> This software is provided "as is" without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.</p>
            <p><b>AI ASSISTANCE:</b> Some information, algorithms, and code structures in this application may have been generated or improved with the assistance of AI language models, including Gemini by Google DeepMind and RelaGen V2 by Versys Research. While AI can be a powerful tool, all critical components have been reviewed and validated by human developers. Final responsibility for the software's behavior rests with the human development team.</p><hr><p align="center">Copyright © Versys Research 2025. All rights reserved.</p>"""
        about_text_edit.setHtml(about_html)
        about_layout.addWidget(about_text_edit)
        self.tabs_panel.addTab(self.about_tab, "About")

    def _connect_signals(self):
        self.refresh_ports_button.clicked.connect(self.serial_worker.list_serial_ports)
        self.connect_button.clicked.connect(self.toggle_connection)
        self.serial_worker.serial_ports_signal.connect(self.update_port_combobox)
        self.serial_worker.connection_status_signal.connect(self.handle_connection_status)
        self.serial_worker.new_data_signal.connect(self.handle_raw_serial_data)
        self.serial_worker.parsed_data_signal.connect(self.process_parsed_data)
        self.serial_worker.log_message_signal.connect(self.log_message)

        # Calibration signals
        self.collect_accel_gyro_button.clicked.connect(self.start_accel_gyro_data_collection)
        self.collect_mag_button.clicked.connect(self.start_mag_data_collection)
        self.process_calib_button.clicked.connect(self.process_and_apply_calib_data)
        self.save_calib_as_button.clicked.connect(self.save_applied_calibration_dialog)
        self.load_calib_button.clicked.connect(self.load_calibration_file_dialog)
        self.clear_calib_buffers_button.clicked.connect(self.clear_calib_data_buffers)
        self.reset_applied_calib_button.clicked.connect(self.reset_applied_calibration_to_zero)

        # AHRS & Visualization signals
        self.tune_ekf_r_button.clicked.connect(self.start_ekf_tune_data_collection)
        self.tune_madgwick_beta_button.clicked.connect(self.start_madgwick_beta_tune_data_collection) # New signal
        self.reset_ahrs_button.clicked.connect(self.reset_ahrs_state_button_clicked)
        self.set_vis_offset_button.clicked.connect(self.set_visualizer_offset)
        self.reset_vis_offset_button.clicked.connect(self.reset_visualizer_offset)
        self.autoscale_mag_checkbox.stateChanged.connect(self.toggle_mag_autoscale)
        self.zoom_fit_mag_button.clicked.connect(self.zoom_fit_mag_plot)
        self.vis_help_button.clicked.connect(self.show_visualizer_help)

        # Benchmark signals
        self.start_benchmark_button.clicked.connect(self.start_static_benchmark)
        self.clear_benchmark_history_button.clicked.connect(self.clear_benchmark_history)

        # Settings signals
        self.save_settings_button.clicked.connect(self.save_app_settings)
        self.reset_settings_button.clicked.connect(self.reset_app_settings)

    def log_message(self, message): # Unchanged
        timestamp = time.strftime("%H:%M:%S")
        if hasattr(self, 'log_area'): 
            self.log_area.append(f"{timestamp} - {message}")
        print(f"{timestamp} - {message}")

    def update_port_combobox(self, ports): # Unchanged
        current_port = self.port_combobox.currentText()
        self.port_combobox.clear()
        self.port_combobox.addItems(ports)
        if current_port in ports:
            self.port_combobox.setCurrentText(current_port)
        elif ports:
            self.port_combobox.setCurrentIndex(0)

    def toggle_connection(self): # Unchanged
        if self.serial_worker.serial_port and self.serial_worker.serial_port.is_open:
            self.serial_worker.disconnect_port()
        else:
            port = self.port_combobox.currentText()
            if not port:
                QMessageBox.warning(self, "Connection Error", "Please select a COM port.")
                return
            baudrate = self.settings_manager.get("default_baudrate")
            self.serial_worker.connect_to_port(port, baudrate)
            self._reset_ahrs_with_current_params() 

    def handle_connection_status(self, port_name, is_connected): # Updated to handle more collection states
        if is_connected:
            self.connect_button.setText("Disconnect")
            self.port_combobox.setEnabled(False)
            self.refresh_ports_button.setEnabled(False)
            self.status_label_connection.setText(f"Status: Connected to {port_name}")
            self.last_data_timestamp_ms = None
        else:
            self.connect_button.setText("Connect")
            self.port_combobox.setEnabled(True)
            self.refresh_ports_button.setEnabled(True)
            self.status_label_connection.setText(f"Status: Disconnected (Port: {port_name})")
            
            interrupted_action = None
            if self.calibration_state == "benchmarking_static": interrupted_action = "Benchmark"
            elif self.calibration_state == "collecting_for_ekf_tune": interrupted_action = "EKF Tuning"
            elif self.calibration_state == "collecting_for_madgwick_beta_tune": interrupted_action = "Madgwick Beta Tuning"
            elif "collecting" in self.calibration_state: interrupted_action = "Data collection"
            
            if interrupted_action:
                self.log_message(f"{interrupted_action} interrupted by disconnection.")
                self.calibration_state = "idle"

        self._update_button_states()
        self.update_calibration_status_label()


    def handle_raw_serial_data(self, line): # Unchanged
        self.processing_start_time = time.perf_counter()

    def process_parsed_data(self, parsed_data): 
        if self.processing_start_time is not None:
            processing_end_time = time.perf_counter()
            self.processing_delay_ms = (processing_end_time - self.processing_start_time) * 1000.0
            self.processing_start_time = None
        else:
            self.processing_delay_ms = 0.0

        ts_ms = parsed_data['timestamp_ms']
        if parsed_data['accel'] is not None: self.data_buffer['accel'] = parsed_data['accel']
        if parsed_data['gyro'] is not None: self.data_buffer['gyro'] = parsed_data['gyro']
        if parsed_data['mag'] is not None: self.data_buffer['mag'] = parsed_data['mag']
        self.data_buffer['timestamp_ms'] = ts_ms

        # Data collection logic
        if self.calibration_state == "collecting_gyro_accel":
            if parsed_data['accel'] is not None and parsed_data['gyro'] is not None:
                max_samples = self.settings_manager.get("accel_gyro_calib_samples")
                if len(self.calib_buffers['gyro']) < max_samples:
                    self.calib_buffers['accel'].append(parsed_data['accel'])
                    self.calib_buffers['gyro'].append(parsed_data['gyro'])
                    count = len(self.calib_buffers['gyro'])
                    if count % 50 == 0 or count == 1:
                        self.log_message(f"Collecting Accel/Gyro data: {count}/{max_samples}")
                    if count >= max_samples:
                        self.log_message("Accel/Gyro static data collection complete.")
                        self.accel_gyro_data_collected_flag = True
                        self.calibration_state = "idle"
                        self._update_button_states()
        elif self.calibration_state == "collecting_mag":
            if parsed_data['mag'] is not None:
                max_samples = self.settings_manager.get("mag_calib_samples")
                if len(self.calib_buffers['mag']) < max_samples:
                    self.calib_buffers['mag'].append(parsed_data['mag'])
                    self.mag_calib_collection_buffer_np = np.vstack([self.mag_calib_collection_buffer_np, parsed_data['mag']])
                    count = len(self.calib_buffers['mag'])
                    if count % 50 == 0 or count == 1:
                        self.log_message(f"Collecting Magnetometer data: {count}/{max_samples}")
                    if count >= max_samples:
                        self.log_message("Magnetometer motion data collection complete.")
                        self.mag_data_collected_flag = True
                        self.calibration_state = "idle"
                        self._update_button_states()
        elif self.calibration_state == "collecting_for_ekf_tune":
            max_samples = self.settings_manager.get("ekf_tune_samples")
            
            if parsed_data['accel'] is not None and len(self.ekf_tune_buffer['accel']) < max_samples:
                acc_cal = parsed_data['accel'] - self.calibration_params["accel_offset"]
                self.ekf_tune_buffer['accel'].append(acc_cal)
                current_accel_count = len(self.ekf_tune_buffer['accel'])
                if current_accel_count % 50 == 0 or current_accel_count == 1:
                    self.log_message(f"EKF Tune: Accel samples {current_accel_count}/{max_samples}")

            if parsed_data['mag'] is not None and len(self.ekf_tune_buffer['mag']) < max_samples:
                mag_cal = np.dot(self.calibration_params["mag_matrix"], (parsed_data['mag'] - self.calibration_params["mag_offset"]))
                self.ekf_tune_buffer['mag'].append(mag_cal)
                current_mag_count = len(self.ekf_tune_buffer['mag'])
                if current_mag_count % 50 == 0 or current_mag_count == 1:
                    self.log_message(f"EKF Tune: Mag samples {current_mag_count}/{max_samples}")
            
            if len(self.ekf_tune_buffer['accel']) >= max_samples and \
               len(self.ekf_tune_buffer['mag']) >= max_samples and \
               self.calibration_state == "collecting_for_ekf_tune":
                self._finish_ekf_tuning()
        elif self.calibration_state == "collecting_for_madgwick_beta_tune": # New collection state
            max_samples = self.settings_manager.get("madgwick_tune_samples")
            if parsed_data['gyro'] is not None and len(self.madgwick_tune_buffer['gyro']) < max_samples:
                self.madgwick_tune_buffer['gyro'].append(parsed_data['gyro']) # Collect raw gyro
                count = len(self.madgwick_tune_buffer['gyro'])
                if count % 50 == 0 or count == 1:
                    self.log_message(f"Madgwick Beta Tune: Gyro samples {count}/{max_samples}")
                if count >= max_samples and self.calibration_state == "collecting_for_madgwick_beta_tune":
                    self._finish_madgwick_beta_tuning()


        # AHRS Update Logic
        if self.ahrs_algorithm is None: return

        if self.data_buffer['accel'] is not None and \
           self.data_buffer['gyro'] is not None and \
           self.data_buffer['timestamp_ms'] is not None:

            current_ts_ms = self.data_buffer['timestamp_ms']
            dt = -1.0
            if self.last_data_timestamp_ms is not None:
                dt = (current_ts_ms - self.last_data_timestamp_ms) / 1000.0

            acc_raw = self.data_buffer['accel']; gyro_raw = self.data_buffer['gyro']; mag_raw = self.data_buffer['mag']
            acc_cal = acc_raw - self.calibration_params["accel_offset"] 

            current_algo_name = self.settings_manager.get("ahrs_algorithm")
            orientation_title = f"Device Orientation ({current_algo_name})"

            if dt > self.settings_manager.get("min_valid_dt_sec") and \
               dt < self.settings_manager.get("max_valid_dt_sec"):
                mag_cal = None
                if mag_raw is not None:
                    mag_cal = np.dot(self.calibration_params["mag_matrix"], (mag_raw - self.calibration_params["mag_offset"]))

                if isinstance(self.ahrs_algorithm, EKF_IMU):
                    self.ahrs_algorithm.predict(gyro_raw, dt) 
                    self.ahrs_algorithm.update_accel(acc_cal)
                    if mag_cal is not None: self.ahrs_algorithm.update_mag(mag_cal)
                elif isinstance(self.ahrs_algorithm, MadgwickAHRS):
                    self.ahrs_algorithm.update_sample_period(dt)
                    gyro_for_madgwick = gyro_raw - self.calibration_params["gyro_offset"] 
                    self.ahrs_algorithm.update(gyro_for_madgwick, acc_cal, mag_cal if mag_cal is not None else np.zeros(3))

                self.orientation_quat_display = self.ahrs_algorithm.get_orientation_quaternion()
                if isinstance(self.ahrs_algorithm, EKF_IMU):
                    gyro_proc_for_plot = gyro_raw - self.ahrs_algorithm.get_gyro_bias()
                else:
                    gyro_proc_for_plot = gyro_raw - self.calibration_params["gyro_offset"]
                accel_proc_for_plot = acc_cal
            else:
                self.orientation_quat_display = np.array([1.0,0.0,0.0,0.0])
                accel_proc_for_plot = acc_cal
                gyro_proc_for_plot = gyro_raw - self.calibration_params["gyro_offset"]
                orientation_title = "Device Orientation (No Update)"

            self.delay_text_item.setHtml(f"<div style='color: lightgray; background-color: rgba(40,40,40,0.7); padding: 2px;'>{orientation_title}<br>Delay: {self.processing_delay_ms:.2f} ms</div>")
            try:
                canvas_width = self.canvas_calib.width()
                text_width = self.delay_text_item.boundingRect().width()
                x_offset_pixels = 10
                y_offset_pixels = 5
                self.delay_text_item.setPos(canvas_width - text_width - x_offset_pixels , y_offset_pixels)
            except Exception: pass

            ts_sec = current_ts_ms / 1000.0
            if np.all(np.isfinite(accel_proc_for_plot)) and np.all(np.isfinite(gyro_proc_for_plot)):
                self._update_timeseries_data_list(self.display_data_timeseries['accel'], ts_sec, accel_proc_for_plot)
                self._update_timeseries_data_list(self.display_data_timeseries['gyro'], ts_sec, gyro_proc_for_plot)
            if self.calibration_state == "benchmarking_static":
                self._process_benchmark_data_collection(accel_proc_for_plot, gyro_proc_for_plot)
            self.last_data_timestamp_ms = current_ts_ms
            self.data_buffer = {'accel': None, 'gyro': None, 'mag': None, 'timestamp_ms': None}

    def _update_timeseries_data_list(self, data_dict_list, timestamp_sec, data_vec_3d): # Unchanged
        limit = self.settings_manager.get("max_plot_points")
        data_dict_list['t'].append(float(timestamp_sec)); data_dict_list['x'].append(float(data_vec_3d[0]))
        data_dict_list['y'].append(float(data_vec_3d[1])); data_dict_list['z'].append(float(data_vec_3d[2]))
        while len(data_dict_list['t']) > limit:
            for key_ in data_dict_list: data_dict_list[key_].pop(0)

    def update_gui_elements(self): # Unchanged
        self.update_calibration_status_label()

    def _update_plots(self): # Unchanged 
        plots_updated = False
        if self.display_data_timeseries['accel']['t']:
            self.accel_lines['x'].set_data(self.display_data_timeseries['accel']['t'], self.display_data_timeseries['accel']['x'])
            self.accel_lines['y'].set_data(self.display_data_timeseries['accel']['t'], self.display_data_timeseries['accel']['y'])
            self.accel_lines['z'].set_data(self.display_data_timeseries['accel']['t'], self.display_data_timeseries['accel']['z'])
            self.ax_accel.relim(); self.ax_accel.autoscale_view(True, True, True)
            plots_updated = True
        if self.display_data_timeseries['gyro']['t']:
            self.gyro_lines['x'].set_data(self.display_data_timeseries['gyro']['t'], self.display_data_timeseries['gyro']['x'])
            self.gyro_lines['y'].set_data(self.display_data_timeseries['gyro']['t'], self.display_data_timeseries['gyro']['y'])
            self.gyro_lines['z'].set_data(self.display_data_timeseries['gyro']['t'], self.display_data_timeseries['gyro']['z'])
            if self.ax_gyro.get_shared_x_axes().joined(self.ax_gyro, self.ax_accel):
                 self.ax_gyro.relim(); self.ax_gyro.autoscale_view(tight=None, scalex=False, scaley=True)
            else: self.ax_gyro.relim(); self.ax_gyro.autoscale_view(True, True, True)
            plots_updated = True
        rescale_mag_plot = False
        if self.calibration_state == "collecting_mag" and self.mag_calib_collection_buffer_np.shape[0] > 0:
            valid_raw_mag = self.mag_calib_collection_buffer_np[np.all(np.isfinite(self.mag_calib_collection_buffer_np), axis=1)]
            if valid_raw_mag.shape[0] > 0: self.mag_scatter_raw_plot._offsets3d = (valid_raw_mag[:,0], valid_raw_mag[:,1], valid_raw_mag[:,2])
            self.mag_scatter_calib_plot._offsets3d = ([],[],[])
            plots_updated = True; rescale_mag_plot = True
        elif self.mag_plot_needs_final_update: 
            self.mag_plot_needs_final_update = False
            raw_mag_buffer = np.array(self.calib_buffers['mag'], dtype=float) 
            if raw_mag_buffer.shape[0] > 0 and np.all(np.isfinite(raw_mag_buffer)):
                 self.mag_scatter_raw_plot._offsets3d = (raw_mag_buffer[:,0], raw_mag_buffer[:,1], raw_mag_buffer[:,2])
            else: self.mag_scatter_raw_plot._offsets3d = ([],[],[])

            if self.final_calibrated_mag_display_data.get('x'):
                calib_mag_np = np.array(list(zip(self.final_calibrated_mag_display_data['x'], self.final_calibrated_mag_display_data['y'], self.final_calibrated_mag_display_data['z'])), dtype=float)
                valid_calib_mag = calib_mag_np[np.all(np.isfinite(calib_mag_np), axis=1)]
                if valid_calib_mag.shape[0] > 0: self.mag_scatter_calib_plot._offsets3d = (valid_calib_mag[:,0],valid_calib_mag[:,1],valid_calib_mag[:,2])
                else: self.mag_scatter_calib_plot._offsets3d = ([],[],[])
            else: self.mag_scatter_calib_plot._offsets3d = ([],[],[])
            plots_updated = True; rescale_mag_plot = True
        if rescale_mag_plot:
            if self.mag_plot_autoscale or not self.mag_plot_scaled_once:
                self.ax_mag_scatter.relim(); self.ax_mag_scatter.autoscale_view(True, True, True)
                self._set_axes_equal_3d(self.ax_mag_scatter)
                if not self.mag_plot_autoscale: self.mag_plot_scaled_once = True
        if self.ahrs_algorithm:
            current_orientation = self.ahrs_algorithm.get_orientation_quaternion()
            if np.all(np.isfinite(current_orientation)) and abs(np.linalg.norm(current_orientation) - 1.0) < 1e-3:
                self._plot_orientation_cube(current_orientation)
                self.ax_orientation.set_visible(True); plots_updated = True
            else: self._plot_orientation_cube(np.array([1.0,0.0,0.0,0.0])); plots_updated = True
        else: self._plot_orientation_cube(np.array([1.0,0.0,0.0,0.0])); plots_updated = True
        if plots_updated:
            try: self.canvas_calib.draw_idle()
            except Exception as e: self.log_message(f"Error drawing canvas: {e}")

    def _set_axes_equal_3d(self, ax, data_points=None): # Unchanged
        if data_points is not None and data_points.shape[0] > 0:
            mins = np.min(data_points, axis=0); maxs = np.max(data_points, axis=0)
            limits = np.array([mins, maxs]).T
        else:
            try: limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            except: return
            if not np.all(np.isfinite(limits)): return
        if not np.all(np.isfinite(limits)): return
        for i in range(3):
            if limits[i,0] == limits[i,1]: limits[i,0] -= 0.5; limits[i,1] += 0.5
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        if not np.isfinite(radius) or radius < 1e-3: radius = 1.0
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius]); ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    def _plot_orientation_cube(self, q_device_to_world): # Unchanged
        q_ref_conj = quaternion_conjugate(self.reference_orientation_quat_display)
        q_display = quaternion_multiply(q_ref_conj, q_device_to_world)
        R_display = quaternion_to_rotation_matrix(q_display)
        if not np.all(np.isfinite(R_display)): return
        rotated_verts_world = self.cube_v @ R_display.T
        if not np.all(np.isfinite(rotated_verts_world)): return
        for i, edge_indices in enumerate(self.cube_edges):
            points_for_edge = rotated_verts_world[list(edge_indices), :]
            if np.all(np.isfinite(points_for_edge)):
                line = self.cube_lines_plot[i]
                line.set_data(points_for_edge[:,0], points_for_edge[:,1]); line.set_3d_properties(points_for_edge[:,2])
        axis_length = 0.5
        body_axes_vectors = np.array([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
        world_axes_endpoints = body_axes_vectors @ R_display.T
        world_origin = np.zeros(3)
        if np.all(np.isfinite(world_axes_endpoints)):
            for i in range(3):
                line = self.coord_lines_plot[i]; axis_endpoint_world = world_axes_endpoints[i, :]
                line.set_data([world_origin[0], axis_endpoint_world[0]], [world_origin[1], axis_endpoint_world[1]])
                line.set_3d_properties([world_origin[2], axis_endpoint_world[2]])

    # --- New Calibration Workflow Methods --- (Unchanged from previous response)
    def start_accel_gyro_data_collection(self):
        if not self._is_serial_connected_and_ready("collect accel/gyro data"): return
        reply = QMessageBox.question(self, "Accel/Gyro Data Collection",
                                     "Place IMU flat and perfectly STILL.\nClick OK to start collecting data.",
                                     QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Ok:
            self.log_message("Starting Accel/Gyro static data collection...")
            self.calibration_state = "collecting_gyro_accel"
            self.calib_buffers['accel'].clear()
            self.calib_buffers['gyro'].clear()
            self.accel_gyro_data_collected_flag = False
            self.calibration_is_processed_and_applied = False 
            self._update_button_states()
            self.update_calibration_status_label()
            duration_approx = self.settings_manager.get('accel_gyro_calib_samples') * (1/ (1000/self.settings_manager.get('gui_update_interval_ms')) ) 
            self.log_message(f"Keep sensor stationary for ~{self.settings_manager.get('accel_gyro_calib_samples') / 50.0 :.1f} seconds (approx).") 
        else:
            self.log_message("Accel/Gyro data collection cancelled.")

    def start_mag_data_collection(self):
        if not self._is_serial_connected_and_ready("collect magnetometer data"): return
        reply = QMessageBox.question(self, "Magnetometer Data Collection",
                                     "SLOWLY rotate IMU in all directions (figure 8 motion is good).\nClick OK to start.",
                                     QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Ok:
            self.log_message("Starting Magnetometer motion data collection...")
            self.calibration_state = "collecting_mag"
            self.calib_buffers['mag'].clear()
            self.mag_calib_collection_buffer_np = np.empty((0,3))
            self.final_calibrated_mag_display_data = {'x': [], 'y': [], 'z': []} 
            self.mag_data_collected_flag = False
            self.calibration_is_processed_and_applied = False 
            self.mag_plot_needs_final_update = False
            self.mag_plot_scaled_once = False
            self.mag_scatter_raw_plot._offsets3d = ([],[],[])
            self.mag_scatter_calib_plot._offsets3d = ([],[],[])
            self._update_button_states()
            self.update_calibration_status_label()
        else:
            self.log_message("Magnetometer data collection cancelled.")

    def process_and_apply_calib_data(self):
        if not self.accel_gyro_data_collected_flag or not self.mag_data_collected_flag:
            QMessageBox.warning(self, "Incomplete Data", "Please collect both Accel/Gyro and Magnetometer data first.")
            return

        self.log_message("Processing collected calibration data...")
        # Accel/Gyro processing
        if len(self.calib_buffers['gyro']) > 10 and len(self.calib_buffers['accel']) > 10:
            gyro_np_raw = np.array(self.calib_buffers['gyro'], dtype=float)
            accel_np_raw = np.array(self.calib_buffers['accel'], dtype=float)
            if np.all(np.isfinite(gyro_np_raw)) and np.all(np.isfinite(accel_np_raw)):
                self.calibration_params["gyro_offset"] = np.mean(gyro_np_raw, axis=0)
                accel_mean_raw = np.mean(accel_np_raw, axis=0)
                g = self.settings_manager.get("gravity")
                accel_offset_candidate = accel_mean_raw.copy()
                dom_axis_idx = np.argmax(np.abs(accel_mean_raw))
                gravity_sign = np.sign(accel_mean_raw[dom_axis_idx])
                expected_gravity_vector = np.zeros(3)
                expected_gravity_vector[dom_axis_idx] = gravity_sign * g
                
                if abs(np.linalg.norm(accel_mean_raw) - g) < g * 0.2 and \
                   abs(accel_mean_raw[dom_axis_idx]) > g * 0.8 : 
                    accel_offset_candidate[dom_axis_idx] -= gravity_sign * g
                    self.calibration_params["accel_offset"] = accel_offset_candidate
                    self.log_message(f"Adjusted Accel offset on axis {['X','Y','Z'][dom_axis_idx]} for gravity ({gravity_sign*g:.2f}).")
                else:
                    self.calibration_params["accel_offset"] = accel_mean_raw 
                    self.log_message(f"Accel mean {accel_mean_raw}, dominant axis {['X','Y','Z'][dom_axis_idx]} not strongly aligned with gravity. Using raw mean as offset.")
                self.log_message(f"Processed Gyro Offset: {self.calibration_params['gyro_offset']}")
                self.log_message(f"Processed Accel Offset: {self.calibration_params['accel_offset']}")
            else: self.log_message("Warning: Non-finite Accel/Gyro calib data during processing.")
        else: self.log_message("Warning: Insufficient Accel/Gyro calib data for processing.")

        # Magnetometer processing
        if len(self.calib_buffers['mag']) > 20:
            mag_data_np_raw = np.array(self.calib_buffers['mag'], dtype=float)
            if np.all(np.isfinite(mag_data_np_raw)):
                h_offset, s_matrix = fit_ellipsoid(mag_data_np_raw)
                if h_offset is not None and s_matrix is not None:
                    self.calibration_params["mag_offset"] = h_offset
                    self.calibration_params["mag_matrix"] = s_matrix
                    self.log_message(f"Processed Mag Hard Iron: {h_offset}")
                    self.log_message(f"Processed Mag Soft Iron:\n{s_matrix}")
                    calibrated_mag_pts = [np.dot(s_matrix, (raw_pt - h_offset)) for raw_pt in mag_data_np_raw]
                    calib_mag_np = np.array(calibrated_mag_pts, dtype=float)
                    if calib_mag_np.shape[0] > 0 and np.all(np.isfinite(calib_mag_np)):
                        self.final_calibrated_mag_display_data = {'x': calib_mag_np[:,0].tolist(),
                                                                  'y': calib_mag_np[:,1].tolist(),
                                                                  'z': calib_mag_np[:,2].tolist()}
                else: self.log_message("Magnetometer fitting failed during processing. Mag params not updated.")
            else: self.log_message("Warning: Non-finite Mag calib data during processing.")
        else: self.log_message("Warning: Insufficient Mag calib data for processing.")

        self.calibration_is_processed_and_applied = True
        self._reset_ahrs_with_current_params() 
        self.log_message("Calibration data processed and applied to AHRS.")
        QMessageBox.information(self, "Processing Complete", "Calibration parameters calculated and applied to AHRS.")
        self.mag_plot_needs_final_update = True 
        self._update_button_states()
        self.update_calibration_status_label()

    def save_applied_calibration_dialog(self):
        if not self.calibration_is_processed_and_applied:
            QMessageBox.warning(self, "No Processed Calibration", "Please process collected data or load a calibration file first.")
            return

        default_filename = self.settings_manager.get("calib_file")
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Applied Calibration As...", default_filename,
                                                  "JSON Files (*.json);;All Files (*)")
        if fileName:
            if self.save_calibration_file(filepath=fileName):
                self.log_message(f"Applied calibration saved to {fileName}")
                QMessageBox.information(self, "Save Successful", f"Applied calibration saved to {fileName}")
            else:
                self.log_message(f"Failed to save applied calibration to {fileName}")
                QMessageBox.critical(self, "Save Error", "Failed to save calibration. Check log.")

    def load_calibration_file_dialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Load Calibration File", "", "JSON Files (*.json);;All Files (*)")
        if fileName:
            if self.load_calibration_file(fileName):
                QMessageBox.information(self, "Load Successful", f"Calibration loaded from {fileName}")
            else:
                QMessageBox.warning(self, "Load Failed", f"Could not load calibration from {fileName}. Check logs.")

    def clear_calib_data_buffers(self):
        reply = QMessageBox.question(self, "Clear Data Buffers",
                                     "This will clear any un-processed Accel/Gyro and Mag data collected. Are you sure?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._clear_calib_buffers_and_flags(clear_processed_flag=False) 
            self.log_message("Calibration data buffers cleared.")
            self._update_button_states()
            self.update_calibration_status_label()

    def reset_applied_calibration_to_zero(self):
        reply = QMessageBox.question(self, "Reset Applied Calibration",
                                     "This will reset all active calibration parameters (offsets to zero, matrix to identity) and clear data buffers. Are you sure?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.calibration_params["accel_offset"] = np.zeros(3, dtype=float)
            self.calibration_params["gyro_offset"] = np.zeros(3, dtype=float)
            self.calibration_params["mag_offset"] = np.zeros(3, dtype=float)
            self.calibration_params["mag_matrix"] = np.identity(3, dtype=float)
            self.calibration_is_processed_and_applied = True 
            self._clear_calib_buffers_and_flags(clear_processed_flag=False) 
            
            self.final_calibrated_mag_display_data = {'x': [], 'y': [], 'z': []} 
            self.mag_plot_needs_final_update = True

            self._reset_ahrs_with_current_params()
            self.log_message("Applied calibration parameters reset to zero/identity.")
            self._update_button_states()
            self.update_calibration_status_label()

    # --- EKF Tuning Methods --- (Unchanged from previous response)
    def start_ekf_tune_data_collection(self):
        if not self._is_serial_connected_and_ready("tune EKF R-values"): return
        if self.settings_manager.get("ahrs_algorithm") != "EKF":
            QMessageBox.warning(self, "Wrong Algorithm", "EKF R-Value tuning is only applicable when EKF is the selected AHRS algorithm.")
            return

        reply = QMessageBox.question(self, "EKF R-Value Tuning",
                                     "Place IMU perfectly STILL.\nThis will collect data to estimate measurement noise for Accel & Mag.\nClick OK to start.",
                                     QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Ok:
            self.log_message("Starting EKF R-value tuning data collection...")
            self.calibration_state = "collecting_for_ekf_tune"
            self.ekf_tune_buffer['accel'].clear()
            self.ekf_tune_buffer['mag'].clear()
            self._update_button_states()
            self.update_calibration_status_label()
            duration_approx = self.settings_manager.get('ekf_tune_samples') / 50.0 
            self.log_message(f"Keep sensor stationary for ~{duration_approx:.1f} seconds (approx).")
        else:
            self.log_message("EKF R-value tuning cancelled.")

    def _finish_ekf_tuning(self):
        self.log_message("EKF R-tuning data collection complete. Analyzing...")
        self.calibration_state = "idle" 

        if len(self.ekf_tune_buffer['accel']) < 20 or len(self.ekf_tune_buffer['mag']) < 20: 
            self.log_message("Not enough data collected for EKF R-tuning. Aborting.")
            QMessageBox.warning(self, "Tuning Error", "Insufficient data collected for EKF R-tuning.")
            self._update_button_states()
            self.update_calibration_status_label()
            return

        accel_data = np.array(self.ekf_tune_buffer['accel'])
        mag_data = np.array(self.ekf_tune_buffer['mag'])

        var_accel = np.mean(np.var(accel_data, axis=0))
        var_mag = np.mean(np.var(mag_data, axis=0))

        min_r_accel = self.settings_manager.get("min_ekf_r_accel_noise")
        min_r_mag = self.settings_manager.get("min_ekf_r_mag_noise")

        tuned_r_accel = max(min_r_accel, var_accel if np.isfinite(var_accel) else min_r_accel)
        tuned_r_mag = max(min_r_mag, var_mag if np.isfinite(var_mag) else min_r_mag)

        self.log_message(f"Tuned EKF R_accel: {tuned_r_accel:.6f} (from raw var: {var_accel:.6f})")
        self.log_message(f"Tuned EKF R_mag: {tuned_r_mag:.6f} (from raw var: {var_mag:.6f})")

        self.settings_manager.set("ekf_r_accel_noise", str(tuned_r_accel))
        self.settings_manager.set("ekf_r_mag_noise", str(tuned_r_mag))

        self.save_app_settings() 
        QMessageBox.information(self, "EKF Tuning Complete",
                                f"Suggested EKF R-values set and saved:\n"
                                f"R_accel_noise: {tuned_r_accel:.6f}\n"
                                f"R_mag_noise: {tuned_r_mag:.6f}")
        self._update_button_states()
        self.update_calibration_status_label()

    # --- Madgwick Beta Tuning Methods ---
    def start_madgwick_beta_tune_data_collection(self):
        if not self._is_serial_connected_and_ready("tune Madgwick Beta"): return
        if self.settings_manager.get("ahrs_algorithm") != "Madgwick":
            QMessageBox.warning(self, "Wrong Algorithm", "Madgwick Beta tuning is only applicable when Madgwick is the selected AHRS algorithm.")
            return

        reply = QMessageBox.question(self, "Madgwick Beta Tuning",
                                     "Place IMU perfectly STILL.\nThis will collect gyroscope data to suggest a Beta value.\nClick OK to start.",
                                     QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Ok:
            self.log_message("Starting Madgwick Beta tuning data collection...")
            self.calibration_state = "collecting_for_madgwick_beta_tune"
            self.madgwick_tune_buffer['gyro'].clear()
            self._update_button_states()
            self.update_calibration_status_label()
            duration_approx = self.settings_manager.get('madgwick_tune_samples') / 50.0 
            self.log_message(f"Keep sensor stationary for ~{duration_approx:.1f} seconds (approx).")
        else:
            self.log_message("Madgwick Beta tuning cancelled.")

    def _finish_madgwick_beta_tuning(self):
        self.log_message("Madgwick Beta tuning data collection complete. Analyzing...")
        self.calibration_state = "idle"

        if len(self.madgwick_tune_buffer['gyro']) < 20:
            self.log_message("Not enough data collected for Madgwick Beta tuning. Aborting.")
            QMessageBox.warning(self, "Tuning Error", "Insufficient data collected for Madgwick Beta tuning.")
            self._update_button_states()
            self.update_calibration_status_label()
            return

        gyro_raw_data_np = np.array(self.madgwick_tune_buffer['gyro'])
        # Apply current gyro offset calibration
        gyro_calibrated_data_np = gyro_raw_data_np - self.calibration_params["gyro_offset"]
        
        std_dev_gyro_axes = np.std(gyro_calibrated_data_np, axis=0)
        avg_gyro_std_dev = np.mean(std_dev_gyro_axes)

        if not np.isfinite(avg_gyro_std_dev):
            self.log_message("Could not calculate valid average gyro standard deviation for Beta tuning.")
            QMessageBox.warning(self, "Tuning Error", "Invalid gyro data for Beta tuning.")
            self._update_button_states(); self.update_calibration_status_label(); return

        # Heuristic: beta = sqrt(3/4) * gyro_noise_stddev
        suggested_beta = np.sqrt(3.0 / 4.0) * avg_gyro_std_dev
        
        min_beta = self.settings_manager.get("min_madgwick_beta")
        max_beta = self.settings_manager.get("max_madgwick_beta")
        tuned_beta = np.clip(suggested_beta, min_beta, max_beta)

        self.log_message(f"Gyro axis std devs (calib): {std_dev_gyro_axes}. Avg std dev: {avg_gyro_std_dev:.6f}")
        self.log_message(f"Suggested Madgwick Beta: {suggested_beta:.4f}. Clamped & Tuned Beta: {tuned_beta:.4f}")

        self.settings_manager.set("madgwick_beta", str(tuned_beta))
        self.save_app_settings() # This will save, populate GUI, and apply runtime changes

        QMessageBox.information(self, "Madgwick Beta Tuning Complete",
                                f"Suggested Madgwick Beta value set and saved:\n"
                                f"Beta: {tuned_beta:.4f}\n"
                                f"Observe performance and adjust manually in Settings if needed.")
        self._update_button_states()
        self.update_calibration_status_label()

    # --- AHRS Reset --- (Unchanged from previous response)
    def reset_ahrs_state_button_clicked(self):
        algo_name = self.settings_manager.get('ahrs_algorithm')
        reply = QMessageBox.question(self, f"Reset {algo_name} State",
                                     f"This will reset the internal state (orientation, biases if EKF) of the {algo_name} algorithm. Are you sure?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self._reset_ahrs_with_current_params()


    # --- Helper Methods --- (Unchanged from previous response)
    def _is_serial_connected_and_ready(self, action_name="perform this action"):
        if not (self.serial_worker.serial_port and self.serial_worker.serial_port.is_open):
            QMessageBox.warning(self, "Not Connected", f"Please connect to a serial port to {action_name}.")
            return False
        if self.calibration_state != "idle":
            QMessageBox.warning(self, "Busy", f"Application is busy ({self.calibration_state}). Please wait or cancel current operation.")
            return False
        return True

    def _clear_calib_buffers_and_flags(self, clear_processed_flag=True):
        self.calib_buffers = {'accel': [], 'gyro': [], 'mag': []}
        self.mag_calib_collection_buffer_np = np.empty((0,3))
        self.accel_gyro_data_collected_flag = False
        self.mag_data_collected_flag = False
        if clear_processed_flag:
            self.calibration_is_processed_and_applied = False

        if hasattr(self, 'mag_scatter_raw_plot'): 
            self.mag_scatter_raw_plot._offsets3d = ([],[],[])
            if not clear_processed_flag and self.calibration_is_processed_and_applied : 
                pass
            else: 
                self.final_calibrated_mag_display_data = {'x':[],'y':[],'z':[]}
                if hasattr(self, 'mag_scatter_calib_plot'):
                    self.mag_scatter_calib_plot._offsets3d = ([],[],[])
            self.mag_plot_needs_final_update = True 


    def save_calibration_file(self, filepath=None): 
        calib_file_name = filepath if filepath else self.settings_manager.get("calib_file")
        save_data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.calibration_params.items()}
        try:
            if os.path.exists(calib_file_name) and filepath is None:
                backup_name = f"{os.path.splitext(calib_file_name)[0]}_{time.strftime('%Y%m%d_%H%M%S')}.json.bak"
                shutil.copy2(calib_file_name, backup_name); self.log_message(f"Backed up previous calibration to {backup_name}")

            os.makedirs(os.path.dirname(calib_file_name) or '.', exist_ok=True)
            with open(calib_file_name, 'w') as f: json.dump(save_data, f, indent=4)

            if filepath is not None and filepath != self.settings_manager.get("calib_file"):
                self.settings_manager.set("calib_file", calib_file_name)
                # If changing calib_file setting, it should ideally be saved to app_settings.json
                # self.settings_manager.save_settings() # Be careful about circular calls or too frequent saves
            return True
        except Exception as e: self.log_message(f"Error saving calibration file '{calib_file_name}': {e}"); traceback.print_exc(); return False

    def load_calibration_file(self, filepath=None): 
        calib_file_name = filepath if filepath else self.settings_manager.get("calib_file")
        loaded_successfully = False
        if os.path.exists(calib_file_name):
            try:
                with open(calib_file_name, 'r') as f: loaded = json.load(f)
                valid_load = True
                for key in self.calibration_params.keys():
                    if key not in loaded:
                        valid_load = False
                        self.log_message(f"Key '{key}' missing in calibration file '{calib_file_name}'.")
                        break
                
                if valid_load:
                    for key in self.calibration_params.keys():
                        self.calibration_params[key] = np.array(loaded[key], dtype=float)
                    
                    self.log_message(f"Calibration loaded from {calib_file_name}")
                    self.calibration_is_processed_and_applied = True
                    self._clear_calib_buffers_and_flags(clear_processed_flag=False) 

                    raw_mag_for_calib_display = np.array(self.calib_buffers['mag'], dtype=float) 
                    if raw_mag_for_calib_display.shape[0] > 20 and np.all(np.isfinite(raw_mag_for_calib_display)):
                         calibrated_mag_pts = [np.dot(self.calibration_params["mag_matrix"], (raw_pt - self.calibration_params["mag_offset"])) for raw_pt in raw_mag_for_calib_display]
                         calib_mag_np = np.array(calibrated_mag_pts, dtype=float)
                         if calib_mag_np.shape[0]>0 and np.all(np.isfinite(calib_mag_np)):
                            self.final_calibrated_mag_display_data = {'x': calib_mag_np[:,0].tolist(), 'y': calib_mag_np[:,1].tolist(), 'z': calib_mag_np[:,2].tolist()}
                    else: 
                        self.final_calibrated_mag_display_data = {'x': [], 'y': [], 'z': []}
                    self.mag_plot_needs_final_update = True


                    if filepath and filepath != self.settings_manager.get("calib_file"):
                         self.settings_manager.set("calib_file", calib_file_name)

                    self._reset_ahrs_with_current_params()
                    loaded_successfully = True
                else:
                    self.log_message(f"Calibration file '{calib_file_name}' is malformed. Using current/defaults.")

            except Exception as e: self.log_message(f"Error loading calib file '{calib_file_name}': {e}. Using current/defaults.")
        else:
            self.log_message(f"Calibration file '{calib_file_name}' not found. Using current/default calibration params.")
            if not self.calibration_is_processed_and_applied :
                 self.reset_applied_calibration_to_zero() 

        self._update_button_states()
        self.update_calibration_status_label()
        return loaded_successfully


    # --- Visualizer and Benchmark Methods --- (Unchanged from previous response)
    def set_visualizer_offset(self): 
        if np.all(np.isfinite(self.orientation_quat_display)):
            self.reference_orientation_quat_display = self.orientation_quat_display.copy()
            self.log_message("Current orientation set as visualizer zero.")
        else: QMessageBox.warning(self, "Error", "Current orientation invalid.")

    def reset_visualizer_offset(self): 
        self.reference_orientation_quat_display = np.array([1.0, 0.0, 0.0, 0.0])
        self.log_message("Visualizer zero reset.")

    def toggle_mag_autoscale(self, state): 
        self.mag_plot_autoscale = bool(state)
        if self.mag_plot_autoscale: self.mag_plot_scaled_once = False
        self.log_message(f"Mag plot autoscale {'Enabled' if self.mag_plot_autoscale else 'Disabled'}.")

    def zoom_fit_mag_plot(self): 
        self.log_message("Zoom fitting magnetometer plot...")
        data_to_fit = []
        if self.calibration_state == "collecting_mag" and self.mag_calib_collection_buffer_np.shape[0] > 0: data_to_fit.append(self.mag_calib_collection_buffer_np)
        raw_mag_buffer = np.array(self.calib_buffers['mag'], dtype=float)
        if raw_mag_buffer.shape[0]>0: data_to_fit.append(raw_mag_buffer)
        if self.final_calibrated_mag_display_data.get('x'):
            calib_mag_np = np.array(list(zip(self.final_calibrated_mag_display_data['x'], self.final_calibrated_mag_display_data['y'], self.final_calibrated_mag_display_data['z'])), dtype=float)
            if calib_mag_np.shape[0]>0: data_to_fit.append(calib_mag_np)
        if not data_to_fit:
            self.log_message("No magnetometer data available to fit."); self.ax_mag_scatter.relim()
            self.ax_mag_scatter.autoscale_view(True, True, True)
        else:
            all_points = np.vstack(data_to_fit)
            valid_points = all_points[np.all(np.isfinite(all_points), axis=1)]
            if valid_points.shape[0] > 0: self._set_axes_equal_3d(self.ax_mag_scatter, data_points=valid_points)
            else: self.ax_mag_scatter.relim(); self.ax_mag_scatter.autoscale_view(True,True,True)
        self.mag_plot_scaled_once = True; self.canvas_calib.draw_idle()

    def show_visualizer_help(self): 
        help_text = """3D Orientation Visualizer Guide:\n- Cube: Represents the physical IMU device.\n- Colored Arrows: IMU's local coordinate axes (X-Red, Y-Green, Z-Blue).\nThe visualizer shows orientation based on the selected AHRS algorithm.\n"Set Zero" uses current displayed orientation as reference.\n"Reset Zero" removes any reference offset."""
        QMessageBox.information(self, "3D Visualizer Help", help_text)

    def start_static_benchmark(self): 
        if not self._is_serial_connected_and_ready("start static benchmark"): return
        duration = self.settings_manager.get("benchmark_duration_sec")
        reply = QMessageBox.question(self, "Static Benchmark", f"Keep IMU STILL for {duration}s.\nClick OK to start.", QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        if reply == QMessageBox.StandardButton.Ok:
            self.log_message(f"Starting static benchmark for {duration}s...")
            self.calibration_state = "benchmarking_static"
            self.benchmark_buffer = {'accel': [], 'gyro': [], 'start_time': time.monotonic()}
            self._update_button_states()
            self.update_calibration_status_label()

    def _process_benchmark_data_collection(self, accel_proc, gyro_proc): 
        if self.calibration_state == "benchmarking_static":
            self.benchmark_buffer['accel'].append(accel_proc); self.benchmark_buffer['gyro'].append(gyro_proc)
            elapsed_time = time.monotonic() - self.benchmark_buffer['start_time']
            duration = self.settings_manager.get("benchmark_duration_sec")
            self.status_label_calibration.setText(f"Status: Benchmarking ({int(elapsed_time)}/{duration}s)") 
            if elapsed_time >= duration: self._finish_static_benchmark()

    def _finish_static_benchmark(self): 
        self.log_message("Static benchmark complete. Analyzing...")
        accel_data = np.array(self.benchmark_buffer['accel'], dtype=float); gyro_data = np.array(self.benchmark_buffer['gyro'], dtype=float)
        results_header = f"--- Static Benchmark Results ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n"
        results_header += f"AHRS Algorithm: {self.settings_manager.get('ahrs_algorithm')}\n"
        results_content = ""
        if accel_data.shape[0] < 10 or gyro_data.shape[0] < 10: results_content += "Insufficient data collected.\n"
        else:
            gyro_mean = np.mean(gyro_data, axis=0); gyro_std = np.std(gyro_data, axis=0)
            results_content += f"Gyroscope (rad/s - Processed by AHRS/Offsets):\n" 
            results_content += f"  Mean (X,Y,Z): {gyro_mean[0]:.4f}, {gyro_mean[1]:.4f}, {gyro_mean[2]:.4f}\n"
            results_content += f"  Std Dev (X,Y,Z): {gyro_std[0]:.4f}, {gyro_std[1]:.4f}, {gyro_std[2]:.4f} (Noise)\n"
            accel_mean = np.mean(accel_data, axis=0); accel_std = np.std(accel_data, axis=0)
            accel_mags = np.linalg.norm(accel_data, axis=1); accel_mag_mean = np.mean(accel_mags); accel_mag_std = np.std(accel_mags)
            g_val = self.settings_manager.get("gravity")
            results_content += f"Accelerometer (m/s^2 - Calibrated with active offsets):\n" 
            results_content += f"  Mean (X,Y,Z): {accel_mean[0]:.4f}, {accel_mean[1]:.4f}, {accel_mean[2]:.4f}\n"
            results_content += f"  Std Dev (X,Y,Z): {accel_std[0]:.4f}, {accel_std[1]:.4f}, {accel_std[2]:.4f} (Noise)\n"
            results_content += f"  Vector Mag Mean: {accel_mag_mean:.4f} (Expected ~{g_val:.2f})\n"
            results_content += f"  Vector Mag Std Dev: {accel_mag_std:.4f}\n"
        full_result_text = results_header + results_content + "--- End of Benchmark ---\n\n"
        self.benchmark_history.append(full_result_text)
        if len(self.benchmark_history) > self.settings_manager.get("max_benchmark_history"): self.benchmark_history.pop(0)
        self.benchmark_results_area.setText("".join(self.benchmark_history))
        self.benchmark_results_area.verticalScrollBar().setValue(self.benchmark_results_area.verticalScrollBar().maximum())
        self.log_message("Benchmark analysis displayed.")
        self.calibration_state = "idle" 
        self._update_button_states()
        self.update_calibration_status_label()

    def clear_benchmark_history(self): 
        self.benchmark_history.clear(); self.benchmark_results_area.clear()
        self.log_message("Benchmark history cleared.")

    # --- Settings Methods --- (Unchanged from previous response)
    def save_app_settings(self): 
        had_errors = False; previous_ahrs_algo = self.settings_manager.get("ahrs_algorithm")
        for key, entry_widget in self.setting_entries.items():
            value_to_set = "";
            if isinstance(entry_widget, QLineEdit): value_to_set = entry_widget.text()
            elif isinstance(entry_widget, QCheckBox): value_to_set = str(entry_widget.isChecked())
            elif isinstance(entry_widget, QComboBox): value_to_set = entry_widget.currentText()
            else: continue
            if not self.settings_manager.set(key, value_to_set):
                self.log_message(f"Failed to set setting: {key} with value '{value_to_set}'"); had_errors = True
        if not had_errors:
            self.settings_manager.save_settings()
            current_ahrs_algo = self.settings_manager.get("ahrs_algorithm")
            if previous_ahrs_algo != current_ahrs_algo:
                self.log_message(f"AHRS algorithm changed from {previous_ahrs_algo} to {current_ahrs_algo}. Re-initializing.")
                self._initialize_ahrs_algorithm() 
            else: 
                 self.apply_runtime_settings_changes()
            QMessageBox.information(self, "Settings Saved", "Application settings saved.")
        else: QMessageBox.warning(self, "Settings Error", "Some settings were invalid and not saved. Check log.")
        self.populate_settings_gui() 

    def reset_app_settings(self): 
        reply = QMessageBox.question(self, "Reset Settings", "Reset all application settings to default values?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            previous_ahrs_algo = self.settings_manager.get("ahrs_algorithm")
            self.settings_manager.reset_to_defaults()
            current_ahrs_algo = self.settings_manager.get("ahrs_algorithm")
            if previous_ahrs_algo != current_ahrs_algo:
                self._initialize_ahrs_algorithm()
            else:
                self.apply_runtime_settings_changes()
            self.populate_settings_gui()
            self.log_message("Application settings reset to defaults.")

    def apply_runtime_settings_changes(self): 
        if self.ahrs_algorithm:
            if isinstance(self.ahrs_algorithm, EKF_IMU):
                self.ahrs_algorithm._update_noise_matrices() 
                self._reset_ahrs_with_current_params() 
                self.log_message("EKF re-initialized/updated with current settings and calibration.")
            elif isinstance(self.ahrs_algorithm, MadgwickAHRS):
                self.ahrs_algorithm.beta = self.settings_manager.get("madgwick_beta")
                mag_ref_ned = np.array(self.settings_manager.get("mag_ref_vector_ned"), dtype=float)
                norm_mag_ref = np.linalg.norm(mag_ref_ned)
                if norm_mag_ref > 1e-6: self.ahrs_algorithm.mag_ref_normalized_ned = mag_ref_ned / norm_mag_ref
                else: self.ahrs_algorithm.mag_ref_normalized_ned = np.array([1.0, 0.0, 0.0]) 
                self.log_message(f"Madgwick AHRS parameters updated (beta: {self.ahrs_algorithm.beta}).")

        self.gui_update_timer.setInterval(self.settings_manager.get("gui_update_interval_ms"))
        self.plot_update_timer.setInterval(self.settings_manager.get("plot_update_interval_ms"))
        self.log_message(f"GUI update interval: {self.gui_update_timer.interval()}ms, Plot interval: {self.plot_update_timer.interval()}ms")
        if hasattr(self, 'ahrs_status_label'):
             self.ahrs_status_label.setText(f"Current AHRS: {self.settings_manager.get('ahrs_algorithm')}")
        self._update_ahrs_settings_visibility(self.settings_manager.get("ahrs_algorithm"))


    def populate_settings_gui(self): 
        for key, entry_widget in self.setting_entries.items():
            current_val = self.settings_manager.get(key)
            if isinstance(entry_widget, QLineEdit):
                entry_widget.setText(json.dumps(current_val) if isinstance(current_val, list) else str(current_val))
            elif isinstance(entry_widget, QCheckBox): entry_widget.setChecked(bool(current_val))
            elif isinstance(entry_widget, QComboBox): entry_widget.setCurrentText(str(current_val))
        if hasattr(self, 'ahrs_algo_combobox'): self._update_ahrs_settings_visibility(self.ahrs_algo_combobox.currentText())

    def _update_button_states(self):
        is_connected = (self.serial_worker.serial_port is not None and self.serial_worker.serial_port.is_open)
        is_idle = (self.calibration_state == "idle")
        current_algo = self.settings_manager.get("ahrs_algorithm")
        is_ekf = (current_algo == "EKF")
        is_madgwick = (current_algo == "Madgwick")

        # Serial Connection
        self.connect_button.setEnabled(True) 
        self.port_combobox.setEnabled(not is_connected)
        self.refresh_ports_button.setEnabled(not is_connected)

        # Sensor Calibration Group
        self.collect_accel_gyro_button.setEnabled(is_connected and is_idle)
        self.collect_mag_button.setEnabled(is_connected and is_idle)
        self.process_calib_button.setEnabled(is_connected and is_idle and self.accel_gyro_data_collected_flag and self.mag_data_collected_flag)
        self.save_calib_as_button.setEnabled(self.calibration_is_processed_and_applied) 
        self.load_calib_button.setEnabled(is_idle) 
        self.clear_calib_buffers_button.setEnabled(is_idle and (self.accel_gyro_data_collected_flag or self.mag_data_collected_flag))
        self.reset_applied_calib_button.setEnabled(is_idle)

        # AHRS & Visualization Group
        self.tune_ekf_r_button.setEnabled(is_connected and is_idle and is_ekf)
        self.tune_madgwick_beta_button.setEnabled(is_connected and is_idle and is_madgwick)
        
        self.reset_ahrs_button.setEnabled(is_idle) 
        self.set_vis_offset_button.setEnabled(True)
        self.reset_vis_offset_button.setEnabled(True)
        self.autoscale_mag_checkbox.setEnabled(True)
        self.zoom_fit_mag_button.setEnabled(True)
        self.vis_help_button.setEnabled(True)

        # Benchmark Group
        self.start_benchmark_button.setEnabled(is_connected and is_idle)
        self.clear_benchmark_history_button.setEnabled(True)

        # Show/hide algo-specific tune buttons based on current selection (already handled by _update_ahrs_settings_visibility)


    def update_calibration_status_label(self): # Updated
        status_parts = []
        if self.calibration_state != "idle":
            action_text = self.calibration_state.replace('_',' ').capitalize()
            if self.calibration_state == "collecting_gyro_accel":
                action_text += f" ({len(self.calib_buffers['gyro'])}/{self.settings_manager.get('accel_gyro_calib_samples')})"
            elif self.calibration_state == "collecting_mag":
                action_text += f" ({len(self.calib_buffers['mag'])}/{self.settings_manager.get('mag_calib_samples')})"
            elif self.calibration_state == "collecting_for_ekf_tune":
                 max_s = self.settings_manager.get('ekf_tune_samples')
                 a_c = len(self.ekf_tune_buffer['accel'])
                 m_c = len(self.ekf_tune_buffer['mag'])
                 action_text += f" (A:{a_c}/{max_s}, M:{m_c}/{max_s})"
            elif self.calibration_state == "collecting_for_madgwick_beta_tune":
                 action_text += f" (G:{len(self.madgwick_tune_buffer['gyro'])}/{self.settings_manager.get('madgwick_tune_samples')})"
            elif self.calibration_state == "benchmarking_static":
                duration = self.settings_manager.get("benchmark_duration_sec")
                elapsed_time = 0
                if self.benchmark_buffer['start_time']: 
                    elapsed_time = time.monotonic() - self.benchmark_buffer['start_time']
                action_text = f"Benchmarking ({int(elapsed_time)}/{duration}s)"
            status_parts.append(f"Action: {action_text}")

        ag_collected = "AG:Collected" if self.accel_gyro_data_collected_flag else "AG:Needed"
        m_collected = "M:Collected" if self.mag_data_collected_flag else "M:Needed"
        status_parts.append(f"Data: [{ag_collected}, {m_collected}]")

        if self.calibration_is_processed_and_applied:
            status_parts.append("Cal: Applied")
        else:
            status_parts.append("Cal: Not Applied")

        if not status_parts: 
            self.status_label_calibration.setText("Status: Idle") 
        else:
            self.status_label_calibration.setText(" | ".join(status_parts))


    def closeEvent(self, event): # Unchanged
        self.log_message("Close event received. Shutting down...")
        self.gui_update_timer.stop(); self.plot_update_timer.stop()
        self.serial_worker.stop()
        self.log_message("Application exited.")
        event.accept()

    def resizeEvent(self, event): # Unchanged
        super().resizeEvent(event)
        if hasattr(self, 'overlay_view') and hasattr(self, 'overlay_scene'):
            view_rect = self.overlay_view.rect()
            scene_rect_f = QRectF(view_rect)
            self.overlay_view.setSceneRect(scene_rect_f)
            if hasattr(self, 'delay_text_item'):
                try:
                    canvas_width = view_rect.width()
                    text_width = self.delay_text_item.boundingRect().width()
                    x_offset_pixels = 10
                    y_offset_pixels = 5
                    self.delay_text_item.setPos(canvas_width - text_width - x_offset_pixels , y_offset_pixels)
                except Exception: pass 

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = IMUCalibratorApp()
    main_window.show()
    sys.exit(app.exec())