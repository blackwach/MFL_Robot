import time
import os
import math
import threading
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pymodbus.client import ModbusTcpClient
from pymodbus.payload import BinaryPayloadDecoder
from pymodbus.constants import Endian
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread, Lock, Event
import logging
from collections import deque
import traceback
from datetime import datetime, timedelta
import csv
from tkinter import filedialog
from pathlib import Path
import json
import pickle
from copy import deepcopy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class Register:
    def __init__(self, name, address, data_type, enabled=True, description=""):
        self.name = name
        self.address = address
        self.data_type = data_type
        self.enabled = tk.BooleanVar(value=enabled)
        self.description = description
        self.min_value = None
        self.max_value = None
        self.alarm_min = None
        self.alarm_max = None
        self.units = ""
        
    def set_limits(self, min_val=None, max_val=None, alarm_min=None, alarm_max=None, units=""):
        self.min_value = min_val
        self.max_value = max_val
        self.alarm_min = alarm_min
        self.alarm_max = alarm_max
        self.units = units
        
    def get_info(self):
        return {
            "name": self.name,
            "address": self.address,
            "data_type": self.data_type,
            "enabled": self.enabled.get(),
            "description": self.description,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "alarm_min": self.alarm_min,
            "alarm_max": self.alarm_max,
            "units": self.units
        }

class DataArchive:
    """Класс для работы с архивом данных"""
    def __init__(self, archive_dir="archive", max_records=10000):
        self.archive_dir = Path(archive_dir)
        self.max_records = max_records
        self.current_file = None
        self.current_records = 0
        self.lock = Lock()
        
        self.archive_dir.mkdir(exist_ok=True)
        
    def start_new_file(self):
        """Создание нового файла архива"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.archive_dir / f"data_{timestamp}.csv"
        self.current_records = 0
        
        with open(self.current_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "register", "value", "units"])
            
    def add_record(self, timestamp, register_name, value, units=""):
        """Добавление записи в архив"""
        with self.lock:
            if self.current_file is None or self.current_records >= self.max_records:
                self.start_new_file()
                
            with open(self.current_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.isoformat(),
                    register_name,
                    value,
                    units
                ])
            self.current_records += 1
            
    def get_records(self, start_time, end_time, register_names=None):
        """Получение записей из архива за указанный период"""
        records = []
        
        archive_files = sorted(self.archive_dir.glob("data_*.csv"))
        
        for file in archive_files:
            file_date_str = file.stem.split('_')[1]
            file_date = datetime.strptime(file_date_str, "%Y%m%d")
            
            if start_time.date() <= file_date.date() <= end_time.date():
                with open(file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        record_time = datetime.fromisoformat(row['timestamp'])
                        if start_time <= record_time <= end_time:
                            if register_names is None or row['register'] in register_names:
                                records.append({
                                    "timestamp": record_time,
                                    "register": row['register'],
                                    "value": float(row['value']) if '.' in row['value'] else int(row['value']),
                                    "units": row['units']
                                })
        
        return records

class MLModelManager:
    """Менеджер моделей машинного обучения"""
    def __init__(self):
        self.models = {
            'LightGBM': lgb.LGBMClassifier(),
            'RandomForest': RandomForestClassifier(),
            'SVM': SVC(probability=True)
        }
        self.current_model = 'LightGBM'
        self.scaler = StandardScaler()
        self.is_trained = False
        self.classes = ['corrosion', 'crack', 'inclusion', 'pitting', 'no_defect']
        self.feature_names = ['B_field', 'relative_change', 'thickness', 'position', 
                             'history_len', 'mean_prev', 'std_prev', 'range_prev']
        
    def get_model(self):
        return self.models[self.current_model]
    
    def set_model(self, model_name):
        if model_name in self.models:
            self.current_model = model_name
            return True
        return False
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Обучение модели с кросс-валидацией"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)
            
            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('classifier', self.get_model())
            ])
            
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV accuracy: {cv_scores.mean():.2f}")
            
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            print(classification_report(y_test, y_pred))
            
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                        display_labels=self.classes)
            disp.plot()
            plt.title(f'Confusion Matrix ({self.current_model})')
            plt.show()
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Ошибка обучения модели: {str(e)}")
            return False
    
    def predict(self, features):
        """Предсказание типа дефекта"""
        if not self.is_trained:
            return 'unknown'
        
        try:
            scaled_features = self.scaler.transform([features])
            pred = self.get_model().predict(scaled_features)[0]
            return self.classes[pred]
        except Exception as e:
            print(f"Ошибка предсказания: {str(e)}")
            return 'unknown'
    
    def save_model(self, filename):
        """Сохранение модели в файл"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model_name': self.current_model,
                'model': self.get_model(),
                'scaler': self.scaler,
                'classes': self.classes,
                'feature_names': self.feature_names
            }, f)
    
    def load_model(self, filename):
        """Загрузка модели из файла"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.current_model = data['model_name']
                self.models[self.current_model] = data['model']
                self.scaler = data['scaler']
                self.classes = data['classes']
                self.feature_names = data.get('feature_names', self.feature_names)
                self.is_trained = True
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {str(e)}")
            return False

class ThicknessCalculator:
    def __init__(self):
        self.B0 = None
        self.L = 0.005
        self.gamma = 0.003
        self.mu_r0 = 5000
        self.calibration_data = []
        self.defect_threshold = 0.05
        self.min_defect_size = 0.001
        self.adaptive_threshold_window = 50
        self.adaptive_threshold_factor = 2.0
        self.ml_manager = MLModelManager()
        self.surface_profile = None
        self.surface_smoothing = 0.1
        
        self.material_params = {
            'steel': {'mu_r': 5000, 'gamma': 0.003, 'L': 0.005, 'color': '#808080'},
            'aluminum': {'mu_r': 3000, 'gamma': 0.004, 'L': 0.004, 'color': '#C0C0C0'},
            'copper': {'mu_r': 4000, 'gamma': 0.0035, 'L': 0.006, 'color': '#DAA520'}
        }
        self.current_material = 'steel'

    def set_material(self, material_name):
        if material_name in self.material_params:
            self.current_material = material_name
            params = self.material_params[material_name]
            self.mu_r0 = params['mu_r']
            self.gamma = params['gamma']
            self.L = params['L']
            return True
        return False

    def calibrate(self, B0, material_params=None):
        self.B0 = B0
        if material_params:
            self.mu_r0 = material_params.get('mu_r', 5000)
            self.gamma = material_params.get('gamma', 0.003)
            self.L = material_params.get('L', 0.005)

    def add_calibration_point(self, Bm, actual_thickness):
        self.calibration_data.append((Bm, actual_thickness))
        self.calibration_data.sort()

    def calculate_thickness(self, Bm, temperature=20.0, position=None):
        try:
            mu_r = self._apply_temperature_correction(temperature)
            
            if self.calibration_data:
                thickness = self._calculate_from_calibration(Bm)
            else:
                thickness = self._calculate_analytical(Bm, mu_r)
            
            if position is not None and self.surface_profile is not None:
                profile_thickness = np.interp(position, 
                                           self.surface_profile['positions'], 
                                           self.surface_profile['thickness'])
                thickness += profile_thickness
            
            return max(thickness, self.min_defect_size)
            
        except Exception as e:
            print(f"Ошибка расчета толщины: {str(e)}")
            return self.min_defect_size

    def _apply_temperature_correction(self, temperature):
        return self.mu_r0 * (1 + self.gamma * (temperature - 20))

    def _calculate_from_calibration(self, Bm):
        B_values = [p[0] for p in self.calibration_data]
        t_values = [p[1] for p in self.calibration_data]
        
        if len(self.calibration_data) >= 2:
            interp_func = interp1d(B_values, t_values, 
                                 kind='linear', 
                                 fill_value='extrapolate')
            return float(interp_func(Bm))
        else:
            return t_values[0] if self.calibration_data else self.min_defect_size

    def _calculate_analytical(self, Bm, mu_r):
        if self.B0 is None or abs(self.B0) < 1e-10:
            return self.min_defect_size
            
        ratio = abs(Bm) / abs(self.B0)
        if ratio <= 0 or ratio >= 1:
            return self.min_defect_size
            
        thickness = -self.L * math.log(ratio)
        return max(thickness, self.min_defect_size)

    def analyze_defect(self, Bm, temperature=20.0, scan_step=0.01, position=None, history=None):
        result = {
            'has_defect': False,
            'depth': 0,
            'width': 0,
            'severity': 0,
            'type': None,
            'confidence': 0
        }
        
        try:
            if self.B0 is None:
                return result
                
            delta_B = abs(Bm) - abs(self.B0)
            relative_change = delta_B / abs(self.B0)
            
            threshold = self._get_adaptive_threshold(history) if history else self.defect_threshold
            
            if abs(relative_change) > threshold:
                result['has_defect'] = True
                result['severity'] = min(abs(relative_change), 1.0)
                
                thickness = self.calculate_thickness(Bm, temperature, position)
                result['depth'] = thickness
                result['width'] = scan_step * (1 + 10 * abs(relative_change))
                
                features = self._extract_features(Bm, relative_change, thickness, position, history)
                result['type'] = self.ml_manager.predict(features)
                
        except Exception as e:
            print(f"Ошибка анализа дефекта: {str(e)}")
            
        return result

    def _get_adaptive_threshold(self, history):
        if len(history) < self.adaptive_threshold_window:
            return self.defect_threshold
            
        window = history[-self.adaptive_threshold_window:]
        values = [abs(d['Bm'] - abs(self.B0)) / abs(self.B0) for d in window]
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        return max(self.defect_threshold, 
                 mean_val + self.adaptive_threshold_factor * std_dev)

    def _extract_features(self, Bm, relative_change, thickness, position, history):
        features = [
            Bm,
            relative_change,
            thickness,
            position if position is not None else 0,
            len(history) if history else 0
        ]
        
        if history and len(history) > 1:
            prev_values = [d['Bm'] for d in history[-5:]]
            features.extend([
                np.mean(prev_values),
                np.std(prev_values),
                np.max(prev_values) - np.min(prev_values)
            ])
        else:
            features.extend([0, 0, 0])
            
        return features

    def set_surface_profile(self, positions, thickness):
        if len(positions) != len(thickness):
            raise ValueError("Длины массивов positions и thickness должны совпадать")
            
        self.surface_profile = {
            'positions': np.array(positions),
            'thickness': np.array(thickness)
        }
        
        if self.surface_smoothing > 0:
            from scipy.ndimage import gaussian_filter1d
            self.surface_profile['thickness'] = gaussian_filter1d(
                self.surface_profile['thickness'], 
                sigma=self.surface_smoothing
            )

    def train_defect_classifier(self, X, y):
        return self.ml_manager.train(X, y)

    def load_defect_classifier(self, filename):
        return self.ml_manager.load_model(filename)

    def save_defect_classifier(self, filename):
        self.ml_manager.save_model(filename)

class MLTrainingDialog(tk.Toplevel):
    def __init__(self, parent, calculator):
        super().__init__(parent)
        self.title("Обучение модели классификации дефектов")
        self.calculator = calculator
        
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self)
        main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Выберите модель:").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="LightGBM")
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_var, 
                                 values=list(self.calculator.ml_manager.models.keys()))
        model_combo.grid(row=0, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(main_frame, text="Файл с данными:").grid(row=1, column=0, sticky="w")
        self.data_file = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.data_file).grid(row=1, column=1, sticky="we", padx=5)
        ttk.Button(main_frame, text="Обзор...", command=self.browse_file).grid(row=1, column=2, padx=5)
        
        ttk.Label(main_frame, text="Размер тестовой выборки:").grid(row=2, column=0, sticky="w")
        self.test_size = tk.DoubleVar(value=0.2)
        ttk.Entry(main_frame, textvariable=self.test_size, width=5).grid(row=2, column=1, sticky="w", padx=5)
        
        ttk.Label(main_frame, text="Random state:").grid(row=3, column=0, sticky="w")
        self.random_state = tk.IntVar(value=42)
        ttk.Entry(main_frame, textvariable=self.random_state, width=5).grid(row=3, column=1, sticky="w", padx=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=4, columnspan=3, pady=10)
        
        ttk.Button(btn_frame, text="Обучить модель", command=self.train_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отмена", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if filename:
            self.data_file.set(filename)
    
    def train_model(self):
        if not self.data_file.get():
            messagebox.showerror("Ошибка", "Выберите файл с данными")
            return
            
        try:
            data = pd.read_csv(self.data_file.get())
            X = data.drop('target', axis=1).values
            y = data['target'].values
            
            self.calculator.ml_manager.set_model(self.model_var.get())
            
            success = self.calculator.ml_manager.train(
                X, y,
                test_size=self.test_size.get(),
                random_state=self.random_state.get()
            )
            
            if success:
                messagebox.showinfo("Успех", "Модель успешно обучена")
                self.destroy()
            else:
                messagebox.showerror("Ошибка", "Не удалось обучить модель")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обучения модели: {str(e)}")

class MFLScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MFL Сканер Визуализация ПРО (Машин.Обуч.) by Biryukov")
        
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0')
        self.style.configure('TButton', padding=5)
        self.style.configure('Title.TLabel', font=('Helvetica', 12, 'bold'))
        
        self.setup_logging()
        self.initialize_data_files()
        self.initialize_variables()
        self.setup_calculator()
        self.setup_data_structures()
        
        self.create_main_frame()
        self.create_connection_panel()
        self.create_registers_panel()
        self.create_measurement_panel()
        self.create_ml_panel()
        self.create_archive_panel()
        self.create_visualization_panel()
        self.create_status_bar()
        
        self.load_configuration()
        self.load_saved_data()
        
    def setup_logging(self):
        logging.basicConfig(
            filename='mfl_scanner.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()
    
    def initialize_data_files(self):
        self.data_dir = Path("mfl_data")
        try:
            self.data_dir.mkdir(exist_ok=True)
            
            default_files = {
                'calibration.json': {"calibration_points": []},
                'measurements.json': [],
                'material_presets.json': {
                    "Сталь": {"mu": 5000, "Ms": 1.6e6, "alpha": 0.003, "h": 0.002, "default_thickness": 10.0},
                    "Алюминий": {"mu": 3000, "Ms": 1.1e6, "alpha": 0.004, "h": 0.002, "default_thickness": 8.0},
                    "Медь": {"mu": 4000, "Ms": 1.4e6, "alpha": 0.0035, "h": 0.002, "default_thickness": 12.0}
                },
                'settings.json': {
                    "modbus_ip": "192.168.31.169",
                    "modbus_port": 502,
                    "slave_id": 1,
                    "sensor_distance": 0.05,
                    "sample_count": 200,
                    "metal_temp": 20.0,
                    "base_thickness": 10.0,
                    "material_coef": 0.005,
                    "mu": 5000,
                    "hall_min": -670,
                    "hall_max": 670,
                    "hall_min_range": -670.0,
                    "hall_max_range": 670.0
                }
            }
            
            for filename, content in default_files.items():
                path = self.data_dir / filename
                if not path.exists():
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(content, f, indent=4, ensure_ascii=False)
                        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось создать файлы данных:\n{str(e)}")
    
    def initialize_variables(self):
        self.modbus_ip = tk.StringVar(value="192.168.1.100")
        self.modbus_port = tk.IntVar(value=502)
        self.slave_id = tk.IntVar(value=1)
        self.byteorder = tk.StringVar(value="big")
        self.wordorder = tk.StringVar(value="big")
        
        self.sensor_distance = tk.DoubleVar(value=0.05)
        self.sample_count = tk.IntVar(value=200)
        self.metal_temp = tk.DoubleVar(value=20.0)
        
        self.base_thickness = tk.DoubleVar(value=10.0)
        self.material_coef = tk.DoubleVar(value=0.005)
        self.mu = tk.DoubleVar(value=5000)
        self.hall_min = tk.DoubleVar(value=-670)
        self.hall_max = tk.DoubleVar(value=670)
        
        self.hall_min_range = tk.DoubleVar(value=-670.0)
        self.hall_max_range = tk.DoubleVar(value=670.0)
        
        self.calib_hall_value = tk.DoubleVar()
        self.calib_thickness = tk.DoubleVar()
        self.calibration_points = []
        
        self.is_connected = False
        self.is_scanning = False
        self.client = None
        self.auto_update = False
        self.auto_update_interval = tk.IntVar(value=1000)
        
        self.current_material = tk.StringVar(value="Сталь")
        self.available_materials = ["Сталь", "Алюминий", "Медь"]
        
        self.registers = [
            Register("Temperature", 0, "float", True, "Температура датчика"),
            Register("Pressure", 2, "float", True, "Давление в системе"),
            Register("FlowRate", 4, "float", True, "Скорость потока"),
            Register("Status", 6, "int", True, "Статус устройства"),
            Register("Distance", 8, "float", True, "Расстояние"),
            Register("FrontSensor", 10, "float", True, "Передний датчик"),
            Register("RearSensor", 12, "float", True, "Задний датчик")
        ]
        
        self.registers[0].set_limits(min_val=-20, max_val=80, alarm_min=0, alarm_max=60, units="°C")
        self.registers[1].set_limits(min_val=0, max_val=10, alarm_min=1, alarm_max=9, units="bar")
        self.registers[2].set_limits(min_val=0, max_val=100, alarm_min=10, alarm_max=90, units="l/min")
        self.registers[3].set_limits(min_val=0, max_val=255, units="code")
        self.registers[4].set_limits(min_val=0, max_val=10, units="m")
        self.registers[5].set_limits(min_val=-670, max_val=670, units="T")
        self.registers[6].set_limits(min_val=-670, max_val=670, units="T")
    
    def setup_calculator(self):
        self.calculator = ThicknessCalculator()
    
    def setup_data_structures(self):
        self.data_archive = DataArchive()
        self.current_values = {}
        
        self.history = {
            reg.name: {
                'values': deque(maxlen=100),
                'timestamps': deque(maxlen=100)
            } for reg in self.registers
        }
        self.history_lock = Lock()
        
        self.scan_data = {
            'positions': [],
            'B_fields': [],
            'temperatures': [],
            'thicknesses': [],
            'defects': [],
            'timestamps': []
        }
        self.scan_data_lock = threading.Lock()
        
        self.stop_event = Event()
    
    def create_main_frame(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.connection_tab = ttk.Frame(self.notebook)
        self.registers_tab = ttk.Frame(self.notebook)
        self.measurement_tab = ttk.Frame(self.notebook)
        self.ml_tab = ttk.Frame(self.notebook)
        self.archive_tab = ttk.Frame(self.notebook)
        self.visualization_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.connection_tab, text="Подключение")
        self.notebook.add(self.registers_tab, text="Регистры")
        self.notebook.add(self.measurement_tab, text="Измерения")
        self.notebook.add(self.ml_tab, text="Машинное обучение")
        self.notebook.add(self.archive_tab, text="Архив")
        self.notebook.add(self.visualization_tab, text="Визуализация")
    
    def create_connection_panel(self):
        frame = ttk.LabelFrame(self.connection_tab, text="Настройки подключения")
        frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        ttk.Label(frame, text="IP адрес:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.modbus_ip).grid(row=0, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Порт:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.modbus_port).grid(row=1, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Slave ID:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.slave_id).grid(row=2, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Порядок байт:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(frame, textvariable=self.byteorder, values=["big", "little"]).grid(row=3, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Порядок слов:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(frame, textvariable=self.wordorder, values=["big", "little"]).grid(row=4, column=1, sticky="we", padx=5, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=5, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Подключиться", command=self.connect_modbus).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Отключиться", command=self.disconnect_modbus).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Тест подключения", command=self.test_connection).pack(side=tk.LEFT, padx=5)
        
        self.connection_status = ttk.Label(frame, text="Не подключено", foreground="red")
        self.connection_status.grid(row=6, columnspan=2, pady=5)
        
        self.auto_update_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="Автообновление", variable=self.auto_update_var,
                       command=self.toggle_auto_update).grid(row=7, columnspan=2, pady=5)
        
        ttk.Label(frame, text="Интервал (мс):").grid(row=8, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.auto_update_interval).grid(row=8, column=1, sticky="we", padx=5, pady=5)
    
    def create_registers_panel(self):
        frame = ttk.LabelFrame(self.registers_tab, text="Регистры Modbus")
        frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        columns = ("name", "address", "type", "value", "units", "alarms")
        self.registers_tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="browse")
        
        self.registers_tree.heading("name", text="Имя")
        self.registers_tree.heading("address", text="Адрес")
        self.registers_tree.heading("type", text="Тип")
        self.registers_tree.heading("value", text="Значение")
        self.registers_tree.heading("units", text="Ед. изм.")
        self.registers_tree.heading("alarms", text="Аварии")
        
        self.registers_tree.column("name", width=120)
        self.registers_tree.column("address", width=60, anchor='center')
        self.registers_tree.column("type", width=80, anchor='center')
        self.registers_tree.column("value", width=100, anchor='center')
        self.registers_tree.column("units", width=60, anchor='center')
        self.registers_tree.column("alarms", width=80, anchor='center')
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.registers_tree.yview)
        self.registers_tree.configure(yscrollcommand=scrollbar.set)
        
        self.registers_tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")
        
        ttk.Button(btn_frame, text="Обновить все", command=self.read_all_registers).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Обновить выбранный", command=self.read_selected_register).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Добавить регистр", command=self.add_register_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Удалить регистр", command=self.remove_register).pack(side=tk.LEFT, padx=5)
        
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        self.update_registers_table()
    
    def create_measurement_panel(self):
        frame = ttk.LabelFrame(self.measurement_tab, text="Параметры измерения")
        frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        ttk.Label(frame, text="Материал:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        material_combo = ttk.Combobox(frame, textvariable=self.current_material, 
                                    values=self.available_materials, state="readonly")
        material_combo.grid(row=0, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Расстояние между точками (м):").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.sensor_distance).grid(row=1, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Количество точек:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.sample_count).grid(row=2, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Температура металла (°C):").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        ttk.Entry(frame, textvariable=self.metal_temp).grid(row=3, column=1, sticky="we", padx=5, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Начать сканирование", 
                  command=self.start_scanning).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Остановить", 
                  command=self.stop_scanning).pack(side=tk.LEFT, padx=5)
        
        self.progress_var = tk.StringVar(value="Готов к работе")
        ttk.Label(frame, textvariable=self.progress_var).grid(row=5, columnspan=2, pady=5)
        
        self.progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress_bar.grid(row=6, columnspan=2, pady=5, sticky="we")
    
    def create_ml_panel(self):
        frame = ttk.LabelFrame(self.ml_tab, text="Классификация дефектов")
        frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        ttk.Label(frame, text="Текущая модель:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.model_info = ttk.Label(frame, text="Не обучена")
        self.model_info.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Обучить модель", 
                  command=self.show_ml_training_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Загрузить модель", 
                  command=self.load_ml_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сохранить модель", 
                  command=self.save_ml_model).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(frame, text="Порог дефекта:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.defect_threshold = tk.DoubleVar(value=0.05)
        ttk.Entry(frame, textvariable=self.defect_threshold).grid(row=2, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Размер окна адаптации:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        self.adaptive_window = tk.IntVar(value=50)
        ttk.Entry(frame, textvariable=self.adaptive_window).grid(row=3, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(frame, text="Фактор отклонения:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        self.adaptive_factor = tk.DoubleVar(value=2.0)
        ttk.Entry(frame, textvariable=self.adaptive_factor).grid(row=4, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Button(frame, text="Применить параметры", 
                  command=self.apply_ml_parameters).grid(row=5, columnspan=2, pady=10)
    
    def create_archive_panel(self):
        frame = ttk.LabelFrame(self.archive_tab, text="Архив данных")
        frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        param_frame = ttk.Frame(frame)
        param_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(param_frame, text="Начало:").pack(side=tk.LEFT, padx=5)
        self.archive_start = ttk.Entry(param_frame, width=20)
        self.archive_start.pack(side=tk.LEFT, padx=5)
        self.archive_start.insert(0, (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"))
        
        ttk.Label(param_frame, text="Конец:").pack(side=tk.LEFT, padx=5)
        self.archive_end = ttk.Entry(param_frame, width=20)
        self.archive_end.pack(side=tk.LEFT, padx=5)
        self.archive_end.insert(0, datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        ttk.Label(frame, text="Регистры:").pack(anchor="w", padx=5)
        
        self.archive_registers_vars = {}
        registers_frame = ttk.Frame(frame)
        registers_frame.pack(fill=tk.X, pady=5)
        
        for reg in self.registers:
            var = tk.BooleanVar(value=True)
            self.archive_registers_vars[reg.name] = var
            cb = ttk.Checkbutton(registers_frame, text=reg.name, variable=var)
            cb.pack(side=tk.LEFT, padx=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Загрузить данные", command=self.load_archive_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Экспорт в CSV", command=self.export_archive_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Очистить архив", command=self.clear_archive).pack(side=tk.RIGHT, padx=5)
        
        columns = ("timestamp", "register", "value", "units")
        self.archive_tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        self.archive_tree.heading("timestamp", text="Время")
        self.archive_tree.heading("register", text="Регистр")
        self.archive_tree.heading("value", text="Значение")
        self.archive_tree.heading("units", text="Ед. изм.")
        
        self.archive_tree.column("timestamp", width=150)
        self.archive_tree.column("register", width=120)
        self.archive_tree.column("value", width=100)
        self.archive_tree.column("units", width=60)
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.archive_tree.yview)
        self.archive_tree.configure(yscrollcommand=scrollbar.set)
        
        self.archive_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_visualization_panel(self):
        frame = ttk.LabelFrame(self.visualization_tab, text="Визуализация данных")
        frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        
        ttk.Label(frame, text="Выберите данные для отображения:").pack(anchor="w", padx=5)
        
        self.plot_registers_vars = {}
        registers_frame = ttk.Frame(frame)
        registers_frame.pack(fill=tk.X, pady=5)
        
        for reg in self.registers:
            var = tk.BooleanVar(value=True)
            self.plot_registers_vars[reg.name] = var
            cb = ttk.Checkbutton(registers_frame, text=reg.name, variable=var)
            cb.pack(side=tk.LEFT, padx=5)
        
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Показать толщину", 
                  command=lambda: self.plot_data('thickness')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Показать поле", 
                  command=lambda: self.plot_data('field')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Показать дефекты", 
                  command=lambda: self.plot_data('defects')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Очистить график", 
                  command=self.clear_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Экспорт данных", 
                  command=self.export_data).pack(side=tk.RIGHT, padx=5)
        
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, frame)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=True)
    
    def create_status_bar(self):
        self.status_var = tk.StringVar(value="Готов к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_configuration(self):
        config_file = Path("mfl_config.json")
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    
                    self.modbus_ip.set(config.get('modbus_ip', "192.168.1.100"))
                    self.modbus_port.set(config.get('modbus_port', 502))
                    self.slave_id.set(config.get('slave_id', 1))
                    self.byteorder.set(config.get('byteorder', "big"))
                    self.wordorder.set(config.get('wordorder', "big"))
                    
                    if 'registers' in config:
                        self.registers = []
                        for reg_data in config['registers']:
                            reg = Register(
                                reg_data['name'],
                                reg_data['address'],
                                reg_data['data_type'],
                                reg_data.get('enabled', True),
                                reg_data.get('description', "")
                            )
                            reg.set_limits(
                                reg_data.get('min_value'),
                                reg_data.get('max_value'),
                                reg_data.get('alarm_min'),
                                reg_data.get('alarm_max'),
                                reg_data.get('units', "")
                            )
                            self.registers.append(reg)
                    
                    self.update_registers_table()
                    self.status_var.set("Конфигурация загружена")
                    
            except Exception as e:
                self.logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
                messagebox.showerror("Ошибка", f"Не удалось загрузить конфигурацию:\n{str(e)}")
    
    def save_configuration(self):
        config = {
            'modbus_ip': self.modbus_ip.get(),
            'modbus_port': self.modbus_port.get(),
            'slave_id': self.slave_id.get(),
            'byteorder': self.byteorder.get(),
            'wordorder': self.wordorder.get(),
            'registers': [reg.get_info() for reg in self.registers]
        }
        
        try:
            with open("mfl_config.json", 'w') as f:
                json.dump(config, f, indent=4)
                
            self.status_var.set("Конфигурация сохранена")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка сохранения конфигурации: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить конфигурацию:\n{str(e)}")
            return False
    
    def load_saved_data(self):
        try:
            settings_file = self.data_dir / 'settings.json'
            if settings_file.exists():
                with open(settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    self.sensor_distance.set(settings.get("sensor_distance", 0.05))
                    self.sample_count.set(settings.get("sample_count", 200))
                    self.metal_temp.set(settings.get("metal_temp", 20.0))
                    self.base_thickness.set(settings.get("base_thickness", 10.0))
                    self.material_coef.set(settings.get("material_coef", 0.005))
                    self.mu.set(settings.get("mu", 5000))
                    self.hall_min.set(settings.get("hall_min", -670))
                    self.hall_max.set(settings.get("hall_max", 670))
                    self.hall_min_range.set(settings.get("hall_min_range", -670.0))
                    self.hall_max_range.set(settings.get("hall_max_range", 670.0))
            
            calib_file = self.data_dir / 'calibration.json'
            if calib_file.exists():
                with open(calib_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "calibration_points" in data:
                        self.calibration_points = data["calibration_points"]
            
            model_file = self.data_dir / 'defect_classifier.pkl'
            if model_file.exists():
                if self.calculator.load_defect_classifier(model_file):
                    self.model_info.config(text="Модель загружена")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось загрузить сохраненные данные:\n{str(e)}")
    
    def save_all_data(self):
        try:
            settings = {
                "sensor_distance": self.sensor_distance.get(),
                "sample_count": self.sample_count.get(),
                "metal_temp": self.metal_temp.get(),
                "base_thickness": self.base_thickness.get(),
                "material_coef": self.material_coef.get(),
                "mu": self.mu.get(),
                "hall_min": self.hall_min.get(),
                "hall_max": self.hall_max.get(),
                "hall_min_range": self.hall_min_range.get(),
                "hall_max_range": self.hall_max_range.get()
            }
            
            with open(self.data_dir / 'settings.json', 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
            
            with open(self.data_dir / 'calibration.json', 'w', encoding='utf-8') as f:
                json.dump({"calibration_points": self.calibration_points}, f, indent=4)
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения данных: {str(e)}")
            messagebox.showerror("Ошибка", f"Не удалось сохранить данные:\n{str(e)}")
    
    def connect_modbus(self):
        if self.is_connected:
            return True
            
        try:
            self.client = ModbusTcpClient(
                host=self.modbus_ip.get(),
                port=self.modbus_port.get(),
                timeout=2
            )
            
            if self.client.connect():
                self.is_connected = True
                self.connection_status.config(text="Подключено", foreground="green")
                self.status_var.set(f"Подключено к {self.modbus_ip.get()}:{self.modbus_port.get()}")
                return True
            else:
                self.connection_status.config(text="Ошибка подключения", foreground="red")
                self.status_var.set("Ошибка подключения")
                return False
                
        except Exception as e:
            self.logger.error(f"Ошибка подключения: {str(e)}")
            self.connection_status.config(text="Ошибка подключения", foreground="red")
            self.status_var.set(f"Ошибка подключения: {str(e)}")
            return False
    
    def disconnect_modbus(self):
        if self.client and self.client.connected:
            self.client.close()
        self.is_connected = False
        self.connection_status.config(text="Не подключено", foreground="red")
        self.status_var.set("Отключено от устройства")
    
    def test_connection(self):
        if self.connect_modbus():
            try:
                rr = self.client.read_holding_registers(0, 1, slave = self.slave_id.get())
                if rr.isError():
                    messagebox.showwarning("Тест подключения", 
                                        "Устройство доступно, но чтение регистров не удалось")
                else:
                    messagebox.showinfo("Тест подключения", "Устройство доступно и отвечает!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка тестирования: {str(e)}")
    
    def toggle_auto_update(self):
        if self.auto_update_var.get():
            self.start_auto_update()
        else:
            self.stop_auto_update()
    
    def start_auto_update(self):
        if not self.is_connected:
            messagebox.showwarning("Предупреждение", "Сначала подключитесь к устройству")
            self.auto_update_var.set(False)
            return
            
        self.auto_update = True
        self.auto_update_thread = threading.Thread(target=self.auto_update_loop, daemon=True)
        self.auto_update_thread.start()
        self.status_var.set("Автообновление запущено")
    
    def stop_auto_update(self):
        self.auto_update = False
        if hasattr(self, 'auto_update_thread'):
            self.auto_update_thread.join(timeout=1)
        self.status_var.set("Автообновление остановлено")
    
    def auto_update_loop(self):
        while self.auto_update:
            start_time = time.time()
            self.read_all_registers()
            
            elapsed = time.time() - start_time
            sleep_time = max(0, self.auto_update_interval.get() / 1000 - elapsed)
            time.sleep(sleep_time)
    
    def update_registers_table(self):
        self.registers_tree.delete(*self.registers_tree.get_children())
        
        for reg in self.registers:
            value = self.current_values.get(reg.name, "N/A")
            
            alarms = []
            if reg.alarm_min is not None and isinstance(value, (int, float)) and value < reg.alarm_min:
                alarms.append("MIN")
            if reg.alarm_max is not None and isinstance(value, (int, float)) and value > reg.alarm_max:
                alarms.append("MAX")
                
            alarm_text = ", ".join(alarms) if alarms else "OK"
            
            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)
            
            self.registers_tree.insert("", "end", values=(
                reg.name,
                reg.address,
                reg.data_type,
                value_str,
                reg.units,
                alarm_text
            ))
    
    def read_all_registers(self):
        if not self.is_connected:
            messagebox.showwarning("Предупреждение", "Сначала подключитесь к устройству")
            return
            
        try:
            new_values = {}
            timestamp = datetime.now()
            
            for reg in self.registers:
                if not reg.enabled.get():
                    continue
                    
                try:
                    if reg.data_type == "float":
                        response = self.client.read_holding_registers(
                            reg.address, 2, slave=self.slave_id.get())
                        
                        if response.isError():
                            self.logger.warning(f"Ошибка чтения регистра {reg.name}")
                            continue
                            
                        decoder = BinaryPayloadDecoder.fromRegisters(
                            response.registers,
                            byteorder=Endian.BIG if self.byteorder.get() == "big" else Endian.LITTLE,
                            wordorder=Endian.BIG if self.wordorder.get() == "big" else Endian.LITTLE
                        )
                        value = decoder.decode_32bit_float()
                        
                    elif reg.data_type == "int":
                        response = self.client.read_holding_registers(
                            reg.address, 1, slave=self.slave_id.get())
                            
                        if response.isError():
                            self.logger.warning(f"Ошибка чтения регистра {reg.name}")
                            continue
                            
                        value = response.registers[0]
                        if value > 32767:
                            value -= 65536
                    
                    else:
                        continue
                    
                    new_values[reg.name] = value
                    
                    with self.history_lock:
                        self.history[reg.name]['values'].append(value)
                        self.history[reg.name]['timestamps'].append(timestamp)
                    
                    self.data_archive.add_record(timestamp, reg.name, value, reg.units)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка чтения регистра {reg.name}: {str(e)}")
            
            self.current_values.update(new_values)
            self.update_registers_table()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения регистров: {str(e)}")
            self.status_var.set(f"Ошибка чтения: {str(e)}")
            return False
    
    def read_selected_register(self):
        if not self.is_connected:
            messagebox.showwarning("Предупреждение", "Сначала подключитесь к устройству")
            return
            
        selected = self.registers_tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите регистр для чтения")
            return
            
        item = self.registers_tree.item(selected[0])
        reg_name = item['values'][0]
        
        reg = next((r for r in self.registers if r.name == reg_name), None)
        if not reg:
            return
            
        try:
            timestamp = datetime.now()
            
            if reg.data_type == "float":
                response = self.client.read_holding_registers(
                    reg.address, 2, slave=self.slave_id.get())
                
                if response.isError():
                    self.logger.warning(f"Ошибка чтения регистра {reg.name}")
                    return
                    
                decoder = BinaryPayloadDecoder.fromRegisters(
                    response.registers,
                    byteorder=Endian.BIG if self.byteorder.get() == "big" else Endian.LITTLE,
                    wordorder=Endian.BIG if self.wordorder.get() == "big" else Endian.LITTLE
                )
                value = decoder.decode_32bit_float()
                
            elif reg.data_type == "int":
                response = self.client.read_holding_registers(
                    reg.address, 1, slave=self.slave_id.get())
                    
                if response.isError():
                    self.logger.warning(f"Ошибка чтения регистра {reg.name}")
                    return
                    
                value = response.registers[0]
                if value > 32767:
                    value -= 65536
            
            else:
                return
                
            self.current_values[reg.name] = value
            
            with self.history_lock:
                self.history[reg.name]['values'].append(value)
                self.history[reg.name]['timestamps'].append(timestamp)
            
            self.data_archive.add_record(timestamp, reg.name, value, reg.units)
            
            self.update_registers_table()
            self.status_var.set(f"Регистр {reg.name} обновлен")
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения регистра {reg.name}: {str(e)}")
            self.status_var.set(f"Ошибка чтения {reg.name}: {str(e)}")
    
    def add_register_dialog(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Добавить регистр")
        
        ttk.Label(dialog, text="Имя:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        name_entry = ttk.Entry(dialog)
        name_entry.grid(row=0, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(dialog, text="Адрес:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        address_entry = ttk.Entry(dialog)
        address_entry.grid(row=1, column=1, sticky="we", padx=5, pady=5)
        
        ttk.Label(dialog, text="Тип:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        type_combo = ttk.Combobox(dialog, values=["int", "float"])
        type_combo.grid(row=2, column=1, sticky="we", padx=5, pady=5)
        type_combo.current(0)
        
        ttk.Label(dialog, text="Описание:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        desc_entry = ttk.Entry(dialog)
        desc_entry.grid(row=3, column=1, sticky="we", padx=5, pady=5)
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=4, columnspan=2, pady=10)
        
        ttk.Button(btn_frame, text="Добавить", command=lambda: self.add_register(
            name_entry.get(),
            address_entry.get(),
            type_combo.get(),
            desc_entry.get(),
            dialog
        )).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="Отмена", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def add_register(self, name, address, data_type, description="", dialog=None):
        if not name or not address:
            messagebox.showerror("Ошибка", "Укажите имя и адрес регистра")
            return
            
        try:
            address = int(address)
        except ValueError:
            messagebox.showerror("Ошибка", "Адрес должен быть целым числом")
            return
            
        if any(reg.address == address for reg in self.registers):
            messagebox.showerror("Ошибка", "Регистр с таким адресом уже существует")
            return
            
        new_reg = Register(name, address, data_type, True, description)
        self.registers.append(new_reg)
        
        self.update_registers_table()
        
        self.history[name] = {'values': deque(maxlen=100), 'timestamps': deque(maxlen=100)}
        self.current_values[name] = None
        
        if dialog:
            dialog.destroy()
        
        self.status_var.set(f"Добавлен регистр {name}")
    
    def remove_register(self):
        selected = self.registers_tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите регистр для удаления")
            return
            
        item = self.registers_tree.item(selected[0])
        reg_name = item['values'][0]
        
        for i, reg in enumerate(self.registers):
            if reg.name == reg_name:
                del self.registers[i]
                break
                
        if reg_name in self.history:
            del self.history[reg_name]
            
        if reg_name in self.current_values:
            del self.current_values[reg_name]
            
        self.update_registers_table()
        self.status_var.set(f"Удален регистр {reg_name}")
    
    def load_archive_data(self):
        try:
            start_time = datetime.strptime(self.archive_start.get(), "%Y-%m-%d %H:%M")
            end_time = datetime.strptime(self.archive_end.get(), "%Y-%m-%d %H:%M")
            
            if start_time >= end_time:
                messagebox.showerror("Ошибка", "Начальная дата должна быть раньше конечной")
                return
                
            selected_registers = [
                reg_name for reg_name, var in self.archive_registers_vars.items() 
                if var.get()
            ]
            
            if not selected_registers:
                messagebox.showwarning("Предупреждение", "Выберите хотя бы один регистр")
                return
                
            records = self.data_archive.get_records(start_time, end_time, selected_registers)
            
            self.archive_tree.delete(*self.archive_tree.get_children())
            for record in records:
                value_str = f"{record['value']:.2f}" if isinstance(record['value'], float) else str(record['value'])
                self.archive_tree.insert("", "end", values=(
                    record['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    record['register'],
                    value_str,
                    record['units']
                ))
                
            self.status_var.set(f"Загружено {len(records)} записей")
            
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректный формат даты: {str(e)}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки данных: {str(e)}")
            self.logger.error(f"Ошибка загрузки архивных данных: {str(e)}")
    
    def export_archive_data(self):
        try:
            items = self.archive_tree.get_children()
            if not items:
                messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Сохранить архивные данные"
            )
            
            if not filename:
                return
                
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "register", "value", "units"])
                
                for item in items:
                    values = self.archive_tree.item(item)['values']
                    writer.writerow(values)
                    
            self.status_var.set(f"Данные экспортированы в {filename}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта данных: {str(e)}")
            self.logger.error(f"Ошибка экспорта архивных данных: {str(e)}")
    
    def clear_archive(self):
        if messagebox.askyesno("Подтверждение", "Вы действительно хотите очистить архив данных?"):
            try:
                for file in self.data_archive.archive_dir.glob("data_*.csv"):
                    file.unlink()
                    
                self.archive_tree.delete(*self.archive_tree.get_children())
                self.status_var.set("Архив данных очищен")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка очистки архива: {str(e)}")
                self.logger.error(f"Ошибка очистки архива: {str(e)}")
    
    def start_scanning(self):
        if self.is_scanning:
            messagebox.showwarning("Предупреждение", "Сканирование уже выполняется")
            return
            
        with self.scan_data_lock:
            self.scan_data = {
                'positions': [],
                'B_fields': [],
                'temperatures': [],
                'thicknesses': [],
                'defects': [],
                'timestamps': []
            }
        
        self.is_scanning = True
        self.progress_var.set("Сканирование начато")
        self.progress_bar['value'] = 0
        
        scan_thread = threading.Thread(target=self.scan_thread, daemon=True)
        scan_thread.start()
    
    def scan_thread(self):
        try:
            with ModbusTcpClient(
                host=self.modbus_ip.get(),
                port=self.modbus_port.get(),
                timeout=2
            ) as client:
                
                if not client.connect():
                    self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось подключиться к устройству"))
                    return
                
                distance = 0.0
                step = self.sensor_distance.get()
                
                for i in range(self.sample_count.get()):
                    if not self.is_scanning:
                        break
                    
                    data = self.read_modbus_data(client)
                    if not data:
                        continue
                    
                    Bm = data.get("FrontSensor", 0.0)
                    temp = data.get("Temperature", 20.0)
                    thickness = self.calculate_thickness(Bm, temp)
                    defect = self.calculator.analyze_defect(Bm, temp, step, distance)
                    
                    with self.scan_data_lock:
                        self.scan_data['positions'].append(distance)
                        self.scan_data['B_fields'].append(Bm)
                        self.scan_data['temperatures'].append(temp)
                        self.scan_data['thicknesses'].append(thickness)
                        self.scan_data['defects'].append(defect)
                        self.scan_data['timestamps'].append(datetime.now())
                    
                    self.root.after(0, self.update_progress, i+1, distance, thickness)
                    
                    distance += step
                    time.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Ошибка сканирования: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка сканирования: {str(e)}"))
        finally:
            self.is_scanning = False
            self.root.after(0, self.on_scan_complete)
    
    def update_progress(self, current, position, thickness):
        self.progress_var.set(f"Точка {current}/{self.sample_count.get()} - Позиция: {position:.2f} м")
        self.progress_bar['value'] = (current / self.sample_count.get()) * 100
        self.status_var.set(f"Текущая толщина: {thickness:.2f} мм")
    
    def on_scan_complete(self):
        self.progress_var.set("Сканирование завершено")
        messagebox.showinfo("Готово", "Сканирование успешно завершено")
        self.plot_data('thickness')
    
    def stop_scanning(self):
        self.is_scanning = False
        self.progress_var.set("Сканирование остановлено")
    
    def read_modbus_data(self, client):
        data = {}
        
        try:
            # Чтение расстояния (регистр 0, float)
            rr = client.read_holding_registers(address=0, count=2, slave=self.slave_id.get())
            if not rr.isError():
                decoder = BinaryPayloadDecoder.fromRegisters(
                    rr.registers, 
                    byteorder=Endian.BIG,
                    wordorder=Endian.BIG
                )
                data["Distance"] = decoder.decode_32bit_float()
            
            # Чтение переднего датчика (регистр 2, float)
            rr = client.read_holding_registers(address=2, count=2, slave=self.slave_id.get())
            if not rr.isError():
                decoder = BinaryPayloadDecoder.fromRegisters(
                    rr.registers,
                    byteorder=Endian.BIG,
                    wordorder=Endian.BIG
                )
                data["FrontSensor"] = decoder.decode_32bit_float()

            # Чтение заднего датчика (регистр 4, float)
            rr = client.read_holding_registers(address=4, count=2, slave=self.slave_id.get())
            if not rr.isError():
                decoder = BinaryPayloadDecoder.fromRegisters(
                    rr.registers,
                    byteorder=Endian.BIG,
                    wordorder=Endian.BIG
                )
                data["RearSensor"] = decoder.decode_32bit_float()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Ошибка чтения Modbus: {str(e)}")
            return None
    
    def calculate_thickness(self, hall_value, temperature=20.0):
        try:
            if self.calibration_points and len(self.calibration_points) >= 2:
                sorted_points = sorted(self.calibration_points, key=lambda x: x[0])
                x = [p[0] for p in sorted_points]
                y = [p[1] for p in sorted_points]
                
                thickness = np.interp(hall_value, x, y)
            else:
                temp_coef = 0.003
                normalized_hall = hall_value / 670
                
                if temperature is not None:
                    temp_factor = 1 + temp_coef * (temperature - 20)
                else:
                    temp_factor = 1
                
                thickness = self.base_thickness.get() - (self.material_coef.get() * normalized_hall * temp_factor)
            
            return max(thickness, 0.1)
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета толщины: {str(e)}")
            return 0.1
    
    def show_ml_training_dialog(self):
        MLTrainingDialog(self.root, self.calculator)
    
    def load_ml_model(self):
        filename = filedialog.askopenfilename(
            title="Выберите файл модели",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialdir=str(self.data_dir))
        
        if filename:
            if self.calculator.load_defect_classifier(filename):
                self.model_info.config(text=f"Модель загружена: {self.calculator.ml_manager.current_model}")
                messagebox.showinfo("Успех", "Модель успешно загружена")
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить модель")
    
    def save_ml_model(self):
        filename = filedialog.asksaveasfilename(
            title="Сохранить модель",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")],
            initialdir=str(self.data_dir))
        
        if filename:
            self.calculator.save_defect_classifier(filename)
            messagebox.showinfo("Успех", "Модель успешно сохранена")
    
    def apply_ml_parameters(self):
        self.calculator.defect_threshold = self.defect_threshold.get()
        self.calculator.adaptive_threshold_window = self.adaptive_window.get()
        self.calculator.adaptive_threshold_factor = self.adaptive_factor.get()
        
        messagebox.showinfo("Успех", "Параметры классификации обновлены")
    
    def plot_data(self, data_type=None):
        if data_type is None:
            selected_registers = [
                reg_name for reg_name, var in self.plot_registers_vars.items() 
                if var.get()
            ]
            
            if not selected_registers:
                messagebox.showwarning("Предупреждение", "Выберите хотя бы один регистр")
                return
                
            self.ax.clear()
            
            for reg_name in selected_registers:
                if reg_name not in self.history:
                    continue
                    
                with self.history_lock:
                    timestamps = list(self.history[reg_name]['timestamps'])
                    values = list(self.history[reg_name]['values'])
                    
                if not timestamps:
                    continue
                    
                start_time = timestamps[0]
                x = [(t - start_time).total_seconds() for t in timestamps]
                
                reg = next((r for r in self.registers if r.name == reg_name), None)
                units = reg.units if reg else ""
                
                self.ax.plot(x, values, label=f"{reg_name} ({units})")
            
            self.ax.set_xlabel("Время (сек)")
            self.ax.set_ylabel("Значение")
            self.ax.set_title("История значений регистров")
            self.ax.grid(True)
            self.ax.legend()
        else:
            if not self.scan_data['positions']:
                messagebox.showerror("Ошибка", "Нет данных для визуализации")
                return
            
            with self.scan_data_lock:
                positions = np.array(self.scan_data['positions'])
                
                if data_type == 'thickness':
                    values = np.array(self.scan_data['thicknesses'])
                    ylabel = "Толщина (мм)"
                    title = "Распределение толщины"
                    color = 'blue'
                elif data_type == 'field':
                    values = np.array(self.scan_data['B_fields'])
                    ylabel = "Магнитное поле (Тл)"
                    title = "Распределение магнитного поля"
                    color = 'red'
                elif data_type == 'defects':
                    values = np.array([d['depth']*1000 if d['has_defect'] else 0 
                                     for d in self.scan_data['defects']])
                    ylabel = "Глубина дефекта (мм)"
                    title = "Распределение дефектов"
                    color = 'orange'
                else:
                    return
            
            self.ax.clear()
            
            if data_type == 'defects':
                defect_mask = values > 0
                self.ax.scatter(positions[defect_mask], values[defect_mask], 
                              color=color, label="Дефекты")
            else:
                self.ax.plot(positions, values, color=color, label=ylabel.split(' ')[0])
            
            self.ax.set_xlabel("Позиция (м)")
            self.ax.set_ylabel(ylabel)
            self.ax.set_title(title)
            self.ax.grid(True)
            self.ax.legend()
        
        self.canvas.draw()
        self.status_var.set("График построен")
    
    def clear_plot(self):
        self.ax.clear()
        self.canvas.draw()
        self.status_var.set("График очищен")
    
    def export_data(self):
        if not self.scan_data['positions']:
            messagebox.showerror("Ошибка", "Нет данных для экспорта")
            return
        
        filename = filedialog.asksaveasfilename(
            title="Экспорт данных сканирования",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(self.data_dir))
        
        if not filename:
            return
        
        try:
            with self.scan_data_lock:
                data = {
                    'position': self.scan_data['positions'],
                    'B_field': self.scan_data['B_fields'],
                    'temperature': self.scan_data['temperatures'],
                    'thickness': self.scan_data['thicknesses'],
                    'has_defect': [d['has_defect'] for d in self.scan_data['defects']],
                    'defect_type': [d['type'] for d in self.scan_data['defects']],
                    'timestamp': [t.strftime("%Y-%m-%d %H:%M:%S") for t in self.scan_data['timestamps']]
                }
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
            
            messagebox.showinfo("Успех", f"Данные успешно экспортированы в {filename}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта данных: {str(e)}")
    
    def on_closing(self):
        self.stop_auto_update()
        self.disconnect_modbus()
        self.save_configuration()
        self.save_all_data()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = MFLScannerApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}\n{traceback.format_exc()}")
        messagebox.showerror("Критическая ошибка", f"Произошла критическая ошибка: {str(e)}")