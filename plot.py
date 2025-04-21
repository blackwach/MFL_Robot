import time
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
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
import json


class Register:
    def __init__(self, name, address, data_type):
        self.name = name
        self.address = address
        self.data_type = data_type
        self.enabled = tk.BooleanVar(value=True)
        self.min_value = None
        self.max_value = None

    def set_limits(self, min_val=None, max_val=None):
        self.min_value = min_val
        self.max_value = max_val


class ThicknessCalculator:
    def __init__(self):
        self.material_params = {
            'mu': 5000,         # Магнитная проницаемость
            'Ms': 1.6e6,        # Намагниченность насыщения (А/м)
            'alpha': 0.003,     # Температурный коэффициент
            'h': 0.002          # Расстояние до поверхности (м)
        }
        self.calibration_points = []

    def set_material_params(self, mu, Ms, alpha, h):
        """Установка параметров материала"""
        self.material_params = {
            'mu': mu,
            'Ms': Ms,
            'alpha': alpha,
            'h': h
        }

    def add_calibration_point(self, hall_value, thickness):
        """Добавление точки калибровки"""
        self.calibration_points.append((hall_value, thickness))
        self.calibration_points.sort()

    def calculate(self, hall_value, temperature=20.0):
        """Основная функция расчета"""
        try:
            params = self.material_params
            temp_factor = 1 + params['alpha'] * (temperature - 20)

            if self.calibration_points:
                # Интерполяция по калибровочным точкам
                x = [p[0] for p in self.calibration_points]
                y = [p[1] for p in self.calibration_points]
                return np.interp(hall_value, x, y)
            else:
                # Аналитическая модель
                C = (params['mu'] * params['Ms'] * params['h']) / 2
                return max(C / (abs(hall_value) + 1e-10) * temp_factor, 0.1)

        except Exception as e:
            print(f"Calculation error: {str(e)}")
            return 0.1


class MFLScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MFL Сканер Визуализация ПРО by Biryukov")

        # Инициализация переменных
        self.initialize_variables()
        self.setup_logging()

        # Создаем вкладки
        self.create_tabs()
        
        # Создаем GUI элементы
        self.create_gui_elements()

        # Инициализация калибровочных переменных
        self.calib_hall_value = tk.DoubleVar()
        self.calib_thickness = tk.DoubleVar()
        self.calibration_points = []
        self.last_measurements = []
        self.max_measurements = 10

        # Создаем остальные элементы интерфейса
        self.create_hall_range_settings()
        self.create_calibration_panel()
        self.create_last_measurements_panel()

        # Загружаем сохраненные данные
        self.load_saved_data()    

    def create_tabs(self):
        """Создание вкладок приложения"""
        self.tab_control = ttk.Notebook(self.root)
        
        # Создаем вкладки
        self.connection_tab = ttk.Frame(self.tab_control)
        self.settings_tab = ttk.Frame(self.tab_control)
        self.visuals_tab = ttk.Frame(self.tab_control)
        self.history_tab = ttk.Frame(self.tab_control)
        
        self.tab_control.add(self.connection_tab, text="Modbus TCP")
        self.tab_control.add(self.settings_tab, text="Металл")
        self.tab_control.add(self.visuals_tab, text="Данные")
        self.tab_control.add(self.history_tab, text="История")
    
        self.tab_control.pack(expand=1, fill="both")

    def initialize_variables(self):
        """Инициализация основных переменных"""
        # Modbus конфигурация поключения
        self.modbus_ip = tk.StringVar(value="192.168.31.169")
        self.modbus_port = tk.IntVar(value=502)
        self.slave_id = tk.IntVar(value=1)
        
        # Measurement parameters
        self.sensor_distance = tk.DoubleVar(value=0.05)
        self.sample_count = tk.IntVar(value=200)
        self.metal_temp = tk.DoubleVar(value=20.0)

        # Параметры материала
        self.base_thickness = tk.DoubleVar(value=10.0)
        self.material_coef = tk.DoubleVar(value=0.005)
        self.mu = tk.DoubleVar(value=5000)
        self.hall_min = tk.DoubleVar(value=-670)
        self.hall_max = tk.DoubleVar(value=670)
        
        # Размерность датчика Холла
        self.hall_min_range = tk.DoubleVar(value=-670.0)
        self.hall_max_range = tk.DoubleVar(value=670.0)
        
        # Data storage
        self.scan_data = {
            "distances": [],
            "front_sensor": [],
            "rear_sensor": [],
            "temperatures": [],
            "thickness": [],
            "timestamps": []
        }

        self.is_running = False
        self.client = None
        self.current_values = {}
        self.auto_update_interval = 1.0  # Интервал автообновления в секундах

        # История значений
        self.history = {reg.name: {'values': deque(maxlen=100), 'timestamps': deque(maxlen=100)} 
                       for reg in self.registers}
        self.history_lock = Lock()

        self.byteorder = tk.StringVar(value="BIG")
        self.wordorder = tk.StringVar(value="BIG")

        self.alarm_sound_enabled = tk.BooleanVar(value=True)
        self.alarm_flash_enabled = tk.BooleanVar(value=True)
        self.alarm_flash_state = False
        self.alarm_status = {reg.name: False for reg in self.registers}

        self.stop_event = Event()
        self.auto_update_var = tk.BooleanVar(value=False)

    @property
    def registers(self):
        """Список зарегистрированных регистров"""
        return [
            Register("Расстояние", 0, "float"),
            Register("Передний датчик", 2, "float"),
            Register("Задний датчик", 4, "float"),
            Register("Температура", 6, "float")
        ]

    def setup_logging(self):
        """Настройка журнала логирования"""
        logging.basicConfig(
            filename='mfl_scanner.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()

    def create_gui_elements(self):
        """Создание графического интерфейса"""
        # Настройки соединения
        self.create_connection_panel(self.connection_tab)
        
        # Настройки параметров металла
        self.create_metal_settings_panel(self.settings_tab)
        
        # Регистры Modbus
        self.create_registers_panel(self.visuals_tab)
        
        # Управление данными
        self.create_data_visualization_panel(self.visuals_tab)
        
        # История данных
        self.create_history_panel(self.history_tab)
        
        # Дополнительные элементы управления
        self.create_additional_controls()

    def create_metal_settings_panel(self, parent):
        """Панель параметров металла и датчика"""
        frame = ttk.LabelFrame(parent, text="Параметры металла и датчика")
        frame.pack(padx=10, pady=5, fill="x")

        # Используем уже инициализированные переменные
        ttk.Label(frame, text="Базовая толщина (мм):").grid(row=0, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.base_thickness, width=10).grid(row=0, column=1)

        ttk.Label(frame, text="Магнитная проницаемость:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.mu, width=10).grid(row=1, column=1)

        ttk.Label(frame, text="Диапазон датчика (min):").grid(row=2, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.hall_min, width=10).grid(row=2, column=1)

        ttk.Label(frame, text="Диапазон датчика (max):").grid(row=3, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.hall_max, width=10).grid(row=3, column=1)

        ttk.Label(frame, text="Расстояние до поверхности (мм):").grid(row=4, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.sensor_distance, width=10).grid(row=4, column=1)

        ttk.Button(frame, text="Применить параметры",
                command=self.apply_material_params).grid(row=5, columnspan=2, pady=5)

    def apply_material_params(self):
        """Применение новых параметров"""
        try:
            # Проверяем, что переменные существуют
            if not hasattr(self, 'base_thickness') or not hasattr(self, 'mu'):
                return
            
            # Получаем значения с проверкой
            base_thickness = self.base_thickness.get() if self.base_thickness else 10.0
            mu = self.mu.get() if self.mu else 5000
            
            # Здесь можно добавить валидацию значений
            messagebox.showinfo("Успех", "Параметры успешно применены")
            self.update_thickness_calculation()  # Обновляем расчеты
        except Exception as e:
            messagebox.showerror("Ошибка", f"Неверные параметры: {str(e)}")

    def create_connection_panel(self, parent):
        """Панель настройки соединения с устройством"""
        frame = ttk.LabelFrame(parent, text="Настройки Modbus TCP")
        frame.pack(padx=10, pady=5, fill="x")

        ttk.Label(frame, text="IP адрес:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.modbus_ip).grid(row=0, column=1)

        ttk.Label(frame, text="Порт:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.modbus_port).grid(row=1, column=1)

        ttk.Label(frame, text="Slave ID:").grid(row=2, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.slave_id).grid(row=2, column=1)

        ttk.Button(frame, text="Проверить подключение",
                 command=self.test_connection).grid(row=3, column=0, columnspan=2, pady=5)

    def create_material_settings(self):
        """Панель параметров материала"""
        frame = ttk.LabelFrame(self.root, text="Параметры материала")
        frame.pack(padx=10, pady=5, fill="x")
        
        # Поля ввода параметров
        ttk.Label(frame, text="Магнитная проницаемость (μ):").grid(row=0, column=0)
        self.mu_var = tk.DoubleVar(value=5000)
        ttk.Entry(frame, textvariable=self.mu_var).grid(row=0, column=1)
        
        ttk.Label(frame, text="Намагниченность (А/м):").grid(row=1, column=0)
        self.ms_var = tk.DoubleVar(value=1.6e6)
        ttk.Entry(frame, textvariable=self.ms_var).grid(row=1, column=1)
        
        ttk.Label(frame, text="Темп. коэффициент:").grid(row=2, column=0)
        self.alpha_var = tk.DoubleVar(value=0.003)
        ttk.Entry(frame, textvariable=self.alpha_var).grid(row=2, column=1)
        
        ttk.Label(frame, text="Расстояние (м):").grid(row=3, column=0)
        self.h_var = tk.DoubleVar(value=0.002)
        ttk.Entry(frame, textvariable=self.h_var).grid(row=3, column=1)
        
        ttk.Button(frame, text="Применить", 
                command=self.update_material_params).grid(row=4, columnspan=2)

    def update_material_params(self):
        """Обновление параметров материала"""
        self.calculator.set_material_params(
            self.mu_var.get(),
            self.ms_var.get(),
            self.alpha_var.get(),
            self.h_var.get()
        )
        messagebox.showinfo("Успех", "Параметры материала обновлены")

    def update_calibration_list(self):
        """Обновление списка точек калибровки"""
        if hasattr(self, 'calibration_listbox'):
            self.calibration_listbox.delete(0, tk.END)
            for point in sorted(self.calibration_points, key=lambda x: x[0]):
                self.calibration_listbox.insert(tk.END, f"{point[0]:.1f} → {point[1]:.2f} мм")


    def create_registers_panel(self, parent):
        """Таблица регистров Modbus"""
        frame = ttk.LabelFrame(parent, text="Регистры Modbus")
        frame.pack(padx=10, pady=5, fill="both", expand=True)

        # Treeview для отображения регистров
        columns = ("name", "address", "type", "value")
        self.registers_tree = ttk.Treeview(frame, columns=columns, show="headings")

        # Настройка колонок
        self.registers_tree.heading("name", text="Название")
        self.registers_tree.heading("address", text="Адрес")
        self.registers_tree.heading("type", text="Тип")
        self.registers_tree.heading("value", text="Значение")

        self.registers_tree.column("name", width=150)
        self.registers_tree.column("address", width=80, anchor='center')
        self.registers_tree.column("type", width=80, anchor='center')
        self.registers_tree.column("value", width=100, anchor='center')

        # Добавление полосы прокрутки
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.registers_tree.yview)
        self.registers_tree.configure(yscrollcommand=scrollbar.set)

        self.registers_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Кнопки управления
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)

        ttk.Button(btn_frame, text="Добавить", command=self.add_register).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Удалить", command=self.remove_register).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Обновить", command=self.update_registers_list).pack(side="left", padx=5)

        # Заполнение таблицы начальными данными
        self.update_registers_list()

    def create_data_visualization_panel(self, parent):
        """Фрейм для визуализации данных"""
        frame = ttk.LabelFrame(parent, text="Текущие значения (реальное время)")
        frame.pack(padx=10, pady=5, fill="x")

        # Виджет для отображения текущих значений
        self.values_labels = {}
        self.value_frames = {}

        for i, reg in enumerate(self.registers):
            subframe = ttk.Frame(frame)
            subframe.pack(fill="x", padx=5, pady=2)
            self.value_frames[reg.name] = subframe

            ttk.Label(subframe, text=f"{reg.name} ({reg.data_type}): ").pack(side="left")
            var = tk.StringVar(value="N/A")
            self.values_labels[reg.name] = var
            ttk.Label(subframe, textvariable=var).pack(side="left")
            ttk.Button(subframe, text="Обновить",
                      command=lambda r=reg: self.read_single_register(r)).pack(side="right")
            ttk.Button(subframe, text="График",
                      command=lambda r=reg.name: self.show_history_graph(r)).pack(side="right", padx=5)

    def create_hall_range_settings(self):
        """Панель настройки диапазона датчиков Холла"""
        frame = ttk.LabelFrame(self.root, text="Настройка датчиков Холла")
        frame.pack(padx=10, pady=5, fill="x")
        
        # Минимальное значение
        ttk.Label(frame, text="Минимальное значение:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.hall_min_range, width=10).grid(row=0, column=1, sticky="w")
        
        # Максимальное значение
        ttk.Label(frame, text="Максимальное значение:").grid(row=1, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.hall_max_range, width=10).grid(row=1, column=1, sticky="w")
        
        # Кнопка применения
        ttk.Button(frame, text="Применить диапазон", 
                 command=self.apply_hall_range).grid(row=2, columnspan=2, pady=5)

    def apply_hall_range(self):
        """Применение выбранного диапазона датчиков"""
        try:
            min_val = self.hall_min_range.get()
            max_val = self.hall_max_range.get()
            
            if min_val >= max_val:
                messagebox.showerror("Ошибка", "Минимальное значение должно быть меньше максимального")
                return
            
            # Проверяем, что текущие значения в новом диапазоне
            for point in self.calibration_points:
                if not (min_val <= point[0] <= max_val):
                    messagebox.showerror("Ошибка", 
                                       f"Точка калибровки {point[0]} вне нового диапазона")
                    return
            
            messagebox.showinfo("Успех", "Диапазон датчиков успешно изменен")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка изменения диапазона: {str(e)}")

    def normalize_hall_value(self, value):
        """Нормализация значения датчика к диапазону [-1, 1]"""
        min_val = self.hall_min_range.get()
        max_val = self.hall_max_range.get()
        
        # Ограничиваем значение указанным диапазоном
        clamped_value = max(min(value, max_val), min_val)
        
        # Нормализуем к [-1, 1]
        range_center = (max_val + min_val) / 2
        range_half = (max_val - min_val) / 2
        
        return (clamped_value - range_center) / range_half


    def create_history_panel(self, parent):
        """Панель для работы с историей данных"""
        frame = ttk.LabelFrame(parent, text="История измерений")
        frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Выбор регистров для отображения
        ttk.Label(frame, text="Выберите регистры для отображения:").pack(anchor="w")
        
        self.history_vars = {}
        check_frame = ttk.Frame(frame)
        check_frame.pack(fill="x", pady=5)
        
        for reg in self.registers:
            var = tk.BooleanVar(value=True)
            self.history_vars[reg.name] = var
            cb = ttk.Checkbutton(check_frame, text=reg.name, variable=var)
            cb.pack(side="left", padx=5)
        
        # Параметры графика
        param_frame = ttk.Frame(frame)
        param_frame.pack(fill="x", pady=5)
        
        ttk.Label(param_frame, text="Период (сек):").pack(side="left")
        self.history_period = tk.IntVar(value=60)
        ttk.Entry(param_frame, textvariable=self.history_period, width=8).pack(side="left", padx=5)
        
        ttk.Button(frame, text="Показать график", command=self.show_combined_history).pack(pady=5)
        
        # Кнопка экспорта данных
        ttk.Button(frame, text="Экспорт данных", command=self.export_history_data).pack(pady=5)

    def create_additional_controls(self):
        """Дополнительные кнопки управления"""
        control_frame = ttk.Frame(self.root)
        control_frame.pack(padx=10, pady=5, fill="x")

        ttk.Button(control_frame, text="Старт", command=self.start_scanning).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Стоп", command=self.stop_scanning).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Обновить все", command=self.read_all_registers).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Визуализировать", command=self.visualize_data).pack(side="left", padx=5)
        ttk.Button(control_frame, text="График толщины", command=self.plot_thickness).pack(side="left", padx=5)

        # Автообновление
        self.auto_update_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Автообновление", variable=self.auto_update_var,
                      command=self.toggle_auto_update).pack(side="right", padx=5)

    def setup_logging(self):
        """Настройка журнала логирования"""
        logging.basicConfig(
            filename='mfl_scanner.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger()

    @property
    def registers(self):
        """Список зарегистрированных регистров"""
        return [
            Register("Расстояние", 0, "float"),
            Register("Передний датчик", 2, "float"),
            Register("Задний датчик", 4, "float"),
            Register("Температура", 6, "float")
        ]

    def toggle_auto_update(self):
        """Включение/выключение автообновления"""
        if self.auto_update_var.get():
            self.start_auto_update()
        else:
            self.stop_auto_update()

    def start_scanning(self):
        """Начало процесса сканирования"""
        if self.is_running:
            return

        self.scan_data = {
            "distances": [],
            "front_sensor": [],
            "rear_sensor": [],
            "temperatures": [],
            "thickness": [],
            "timestamps": []
        }

        self.is_running = True
        Thread(target=self.collect_data_thread, daemon=True).start()

    def stop_scanning(self):
        """Остановка процесса сканирования"""
        self.is_running = False
        if self.client and self.client.connected:
            self.client.close()

    def collect_data_thread(self):
        """Поток для сбора данных с Modbus устройства"""
        try:
            self.client = ModbusTcpClient(host=self.modbus_ip.get(),
                                       port=self.modbus_port.get(),
                                       timeout=2)

            if not self.client.connect():
                self.logger.error("Не удалось подключиться к Modbus устройству")
                messagebox.showerror("Ошибка", "Не удалось подключиться к устройству Modbus")
                return

            self.logger.info("Начало сбора данных с Modbus устройства")

            for _ in range(self.sample_count.get()):
                if not self.is_running:
                    break

                data = self.read_modbus_data(self.client)
                if data:
                    self.process_scan_data(data)
                time.sleep(0.1)  # Пауза между опросами

            messagebox.showinfo("Готово", "Сбор данных завершён")

        except Exception as e:
            self.logger.error(f"Ошибка сбора данных: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка сбора данных: {str(e)}")
        finally:
            if self.client and self.client.connected:
                self.client.close()
            self.is_running = False
            self.logger.info("Сбор данных остановлен")

    def process_scan_data(self, data):
        """Обработка и сохранение полученных данных"""
        timestamp = datetime.now()

        if "Расстояние" in data and "Передний датчик" in data:
            self.scan_data["distances"].append(data["Расстояние"])
            self.scan_data["front_sensor"].append(data["Передний датчик"])

            temp = data.get("Температура", self.metal_temp.get())
            thickness = self.calculate_thickness(data["Передний датчик"], temp)
            self.scan_data["thickness"].append(thickness)
            self.scan_data["timestamps"].append(timestamp)

        if "Задний датчик" in data:
            self.scan_data["rear_sensor"].append(data["Задний датчик"])

        if "Температура" in data:
            self.scan_data["temperatures"].append(data["Температура"])

        # Обновляем историю
        with self.history_lock:
            for name, value in data.items():
                if name in self.history:
                    self.history[name]['values'].append(value)
                    self.history[name]['timestamps'].append(timestamp)

    def calculate_thickness(self, hall_value, temperature=20.0):
        """
        Расчет толщины по методике Никитина-Гобова
        с учетом температурной компенсации
        """
        try:
            if None in [self.base_thickness, self.hall_min, self.hall_max]:
                return 0.1
            base_thickness = self.base_thickness.get() if self.base_thickness else 10.0
            hall_min = self.hall_min.get() if self.hall_min else -670
            hall_max = self.hall_max.get() if self.hall_max else 670
            # Параметры материала (примерные значения)
            mu = 5000       # Относительная магнитная проницаемость
            Ms = 1.6e6      # Намагниченность насыщения (А/м)
            h = 0.002       # Расстояние датчик-поверхность (м)
            
            # Температурная коррекция
            alpha = 0.003   # Температурный коэффициент
            temp_factor = 1 + alpha * (temperature - 20)
            
            # Нормировка сигнала
            norm_signal = (hall_value - self.hall_min.get()) / (self.hall_max.get() - self.hall_min.get())
            
            # Основной расчет (упрощенная модель)
            if self.calibration_points:
                # Если есть калибровка, используем интерполяцию
                x = [p[0] for p in self.calibration_points]
                y = [p[1] for p in self.calibration_points]
                thickness = np.interp(hall_value, x, y)
            else:
                # Аналитическое решение для плоской стенки
                C = (mu * Ms * h) / (2 * self.base_thickness.get())
                thickness = (C / (norm_signal + 1e-10)) * temp_factor
                
                # Ограничиваем минимальную толщину
                thickness = max(thickness, 0.1)
                
            return thickness
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета: {str(e)}")
            return 0.1


    def visualize_data(self):
        """Визуализация данных в 3D"""
        if not self.scan_data.get("distances"):
            messagebox.showerror("Ошибка", "Нет данных для визуализации")
            return

        try:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')

            distances = np.array(self.scan_data["distances"])
            front_data = np.array(self.scan_data["front_sensor"])
            base_thickness = self.base_thickness.get()

            # Передний датчик
            ax.plot(distances, np.zeros(len(distances)), front_data, 'g-', alpha=0.3, label='Передний датчик')
            scatter1 = ax.scatter(distances, np.zeros(len(distances)), front_data, 
                                c=front_data, cmap='viridis', s=20)
            
            # Задний датчик
            if "rear_sensor" in self.scan_data and len(self.scan_data["rear_sensor"]) == len(distances):
                rear_data = np.array(self.scan_data["rear_sensor"])
                distances_rear = distances + self.sensor_distance.get()
                ax.plot(distances_rear, np.ones(len(distances)), rear_data, 'b-', alpha=0.3, label='Задний датчик')
                scatter2 = ax.scatter(distances_rear, np.ones(len(distances)), rear_data, 
                                    c=rear_data, cmap='plasma', s=20)

            # Пластина и дефекты
            if self.scan_data.get("thickness"):
                thickness = np.array(self.scan_data["thickness"])

                # Создаем сетку для поверхности
                X = np.linspace(min(distances), max(distances), 50)
                Y = np.linspace(-0.5, 1.5, 50)
                X, Y = np.meshgrid(X, Y)

                # Верхняя поверхность пластины
                ax.plot_surface(X, Y, np.ones_like(X)*base_thickness, 
                              color='#C0C0C0', alpha=0.7, label='Поверхность пластины')

                # Нижняя поверхность пластины
                ax.plot_surface(X, Y, np.zeros_like(X), 
                              color='#A9A9A9', alpha=0.7, label='Нижняя поверхность')

                # Дефекты (показываем только значительные отклонения)
                defect_threshold = base_thickness * 0.95  # 5% отклонение считается дефектом
                defect_mask = thickness < defect_threshold

                if np.any(defect_mask):
                    defect_distances = distances[defect_mask]
                    defect_values = thickness[defect_mask]

                    # Визуализация дефектов
                    ax.scatter(defect_distances, np.zeros(len(defect_distances)), 
                             base_thickness - defect_values, 
                             c='red', s=50, label='Дефекты', marker='x')

                    # Подписи для дефектов
                    for i, (d, v) in enumerate(zip(defect_distances, defect_values)):
                        ax.text(d, 0, base_thickness - v, 
                              f'{base_thickness-v:.2f}mm', color='red')

            # Настройки графика
            ax.set_xlabel('Расстояние (м)', fontsize=12)
            ax.set_ylabel('Положение датчика', fontsize=12)
            ax.set_zlabel('Сигнал (Гаусс)/Толщина (мм)', fontsize=12)
            ax.set_title('3D-визуализация MFL сканирования', fontsize=14)

            # Добавляем цветовые бары для датчиков
            if "rear_sensor" in self.scan_data and len(self.scan_data["rear_sensor"]) == len(distances):
                fig.colorbar(scatter1, ax=ax, label='Передний датчик (Гаусс)', shrink=0.5, aspect=10)
                fig.colorbar(scatter2, ax=ax, label='Задний датчик (Гаусс)', shrink=0.5, aspect=10)
            else:
                fig.colorbar(scatter1, ax=ax, label='Сигнал датчика (Гаусс)', shrink=0.5, aspect=10)

            ax.legend(fontsize=10)
            ax.view_init(elev=25, azim=-45)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Ошибка визуализации: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка при визуализации данных: {str(e)}")

    def show_history_graph(self, register_name):
        """Показ графика исторических данных конкретного регистра"""
        if register_name not in self.history or not self.history[register_name]['values']:
            messagebox.showwarning("Предупреждение", f"Нет данных для регистра {register_name}")
            return

        try:
            # Получаем данные из истории
            values = list(self.history[register_name]['values'])
            timestamps = list(self.history[register_name]['timestamps'])

            # Преобразуем временные метки в относительное время в секундах
            if len(timestamps) > 1:
                time_diff = [(t - timestamps[0]).total_seconds() for t in timestamps]
            else:
                time_diff = [0]

            # Создаем график
            fig, ax = plt.subplots(figsize=(12, 6))

            # Определяем тип данных для форматирования
            reg = self.find_register(register_name)
            if reg and reg.data_type == "float":
                values_fmt = [f"{v:.2f}" for v in values]
                ax.plot(time_diff, values, 'b-', label=f"{register_name}")
            else:
                values_fmt = [str(v) for v in values]
                ax.stem(time_diff, values, 'b-', markerfmt='bo', basefmt=" ", label=f"{register_name}")

            # Настройки графика
            ax.set_xlabel('Время (сек)', fontsize=12)
            ax.set_ylabel(f"Значение ({reg.data_type if reg else 'N/A'})", fontsize=12)
            ax.set_title(f"История измерений: {register_name}", fontsize=14)

            # Добавляем аннотации для крайних значений
            if len(values) > 0:
                max_val = max(values)
                min_val = min(values)
                max_idx = values.index(max_val)
                min_idx = values.index(min_val)

                ax.annotate(f'Max: {values_fmt[max_idx]}', 
                           xy=(time_diff[max_idx], max_val),
                           xytext=(10, 10), textcoords='offset points',
                           arrowprops=dict(arrowstyle="->"))

                ax.annotate(f'Min: {values_fmt[min_idx]}', 
                           xy=(time_diff[min_idx], min_val),
                           xytext=(10, -20), textcoords='offset points',
                           arrowprops=dict(arrowstyle="->"))

            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Ошибка построения графика: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка при построении графика: {str(e)}")

    def show_combined_history(self):
        """Показ комбинированного графика истории для выбранных регистров"""
        try:
            selected_registers = [name for name, var in self.history_vars.items() if var.get()]
            if not selected_registers:
                messagebox.showwarning("Предупреждение", "Не выбраны регистры для отображения")
                return

            period = self.history_period.get()

            fig, ax = plt.subplots(figsize=(12, 6))

            for reg_name in selected_registers:
                if reg_name not in self.history or not self.history[reg_name]['values']:
                    continue

                values = list(self.history[reg_name]['values'])
                timestamps = list(self.history[reg_name]['timestamps'])

                if not values or not timestamps:
                    continue

                # Фильтрация по периоду
                if period > 0:
                    cutoff_time = datetime.now() - timedelta(seconds=period)
                    filtered = [(t, v) for t, v in zip(timestamps, values) if t >= cutoff_time]
                    if not filtered:
                        continue
                    filtered_t, filtered_v = zip(*filtered)
                    time_diff = [(t - filtered_t[0]).total_seconds() for t in filtered_t]
                else:
                    time_diff = [(t - timestamps[0]).total_seconds() for t in timestamps]
                    filtered_v = values

                # Определяем тип данных для форматирования линии
                reg = self.find_register(reg_name)
                if reg and reg.data_type == "float":
                    ax.plot(time_diff, filtered_v, '-', label=reg_name)
                else:
                    ax.stem(time_diff, filtered_v, '-', markerfmt='o', basefmt=" ", label=reg_name)

            ax.set_xlabel('Время (сек)', fontsize=12)
            ax.set_ylabel('Значения', fontsize=12)
            ax.set_title(f'История измерений за последние {period} секунд', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Ошибка построения комбинированного графика: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка при построении графика: {str(e)}")

    def export_history_data(self):
        """Экспорт исторических данных в CSV файл"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Сохранить данные истории"
            )

            if not filename:
                return

            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)

                # Заголовки
                headers = ['Timestamp']
                for reg in self.registers:
                    headers.append(reg.name)

                writer.writerow(headers)

                # Собираем все временные метки
                all_timestamps = set()
                for reg in self.registers:
                    if reg.name in self.history:
                        all_timestamps.update(self.history[reg.name]['timestamps'])

                # Сортируем временные метки
                sorted_timestamps = sorted(all_timestamps)

                # Записываем данные
                for ts in sorted_timestamps:
                    row = [ts.strftime('%Y-%m-%d %H:%M:%S.%f')]

                    for reg in self.registers:
                        # Ищем значение для этой временной метки
                        value = None
                        if reg.name in self.history:
                            for t, v in zip(self.history[reg.name]['timestamps'], 
                                        self.history[reg.name]['values']):
                                if t == ts:
                                    value = v
                                    break

                        row.append(str(value) if value is not None else '')

                    writer.writerow(row)

            messagebox.showinfo("Успех", f"Данные успешно экспортированы в {filename}")

        except Exception as e:
            self.logger.error(f"Ошибка экспорта данных: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка при экспорте данных: {str(e)}")

    def show_combined_history(self):
        """Показ комбинированного графика истории для выбранных регистров"""
        try:
            selected_registers = [name for name, var in self.history_vars.items() if var.get()]
            if not selected_registers:
                messagebox.showwarning("Предупреждение", "Не выбраны регистры для отображения")
                return

            period = self.history_period.get()

            fig, ax = plt.subplots(figsize=(12, 6))

            for reg_name in selected_registers:
                if reg_name not in self.history or not self.history[reg_name]['values']:
                    continue

                values = list(self.history[reg_name]['values'])
                timestamps = list(self.history[reg_name]['timestamps'])

                if not values or not timestamps:
                    continue

                # Фильтрация по периоду
                if period > 0:
                    cutoff_time = datetime.now() - timedelta(seconds=period)  # Исправлено здесь
                    filtered = [(t, v) for t, v in zip(timestamps, values) if t >= cutoff_time]
                    if not filtered:
                        continue
                    filtered_t, filtered_v = zip(*filtered)
                    time_diff = [(t - filtered_t[0]).total_seconds() for t in filtered_t]
                else:
                    time_diff = [(t - timestamps[0]).total_seconds() for t in timestamps]
                    filtered_v = values

                # Определяем тип данных для форматирования линии
                reg = self.find_register(reg_name)
                if reg and reg.data_type == "float":
                    ax.plot(time_diff, filtered_v, '-', label=reg_name)
                else:
                    ax.stem(time_diff, filtered_v, '-', markerfmt='o', basefmt=" ", label=reg_name)

            ax.set_xlabel('Время (сек)', fontsize=12)
            ax.set_ylabel('Значения', fontsize=12)
            ax.set_title(f'История измерений за последние {period} секунд', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Ошибка построения комбинированного графика: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", 
                                 f"Ошибка при построении графика: {str(e)}")

    def find_register(self, name):
        """Поиск регистра по имени"""
        return next((reg for reg in self.registers if reg.name == name), None)

    def plot_calibration_curve(self):
        """Построение калибровочной кривой"""
        if not self.calibration_points:
            messagebox.showerror("Ошибка", "Нет данных калибровки")
            return
        
        x = [p[0] for p in self.calibration_points]
        y = [p[1] for p in self.calibration_points]
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'o-', label="Калибровочная кривая")
        plt.xlabel("Значение датчика Холла")
        plt.ylabel("Толщина (мм)")
        plt.title("Зависимость толщины от показаний датчика")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_thickness(self):
        """Построение графика толщины металла"""
        if not self.scan_data.get("thickness"):
            messagebox.showerror("Ошибка", "Нет данных о толщине")
            return

        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            distances = np.array(self.scan_data["distances"])
            thickness = np.array(self.scan_data["thickness"])
            base_thickness = self.base_thickness.get()

            # Основной график толщины
            ax.plot(distances, thickness, 'r-', linewidth=2, label='Толщина металла')
            ax.scatter(distances, thickness, c='red', s=30)

            # Линия базовой толщины
            ax.axhline(y=base_thickness, color='blue', linestyle='--',
                      label=f'Базовая толщина: {base_thickness:.2f} мм')

            # Выделяем области с дефектами
            defect_threshold = base_thickness * 0.95  # 5% отклонение считается дефектом
            defect_mask = thickness < defect_threshold

            if np.any(defect_mask):
                # Заливка областей с дефектами
                ax.fill_between(distances, thickness, base_thickness,
                              where=defect_mask, color='red', alpha=0.2,
                              label='Области дефектов')

                # Подписи для дефектов
                defect_distances = distances[defect_mask]
                defect_values = thickness[defect_mask]

                for d, v in zip(defect_distances, defect_values):
                    ax.text(d, v, f'{base_thickness-v:.2f}mm',
                           color='red', ha='center', va='bottom')

            ax.set_xlabel('Пройденное расстояние (м)', fontsize=12)
            ax.set_ylabel('Толщина (мм)', fontsize=12)
            ax.set_title('График толщины металла с дефектами', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            plt.tight_layout()
            plt.show()

        except Exception as e:
            self.logger.error(f"Ошибка построения графика толщины: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка при построении графика толщины: {str(e)}")

    def create_calibration_panel(self):
        """Панель калибровки датчика Холла"""
        frame = ttk.LabelFrame(self.root, text="Калибровка датчика Холла")
        frame.pack(padx=10, pady=5, fill="x")

        # Поля ввода для калибровки
        ttk.Label(frame, text="Значение датчика:").grid(row=0, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.calib_hall_value, width=10).grid(row=0, column=1)
        
        ttk.Label(frame, text="Толщина (мм):").grid(row=1, column=0, sticky="e")
        ttk.Entry(frame, textvariable=self.calib_thickness, width=10).grid(row=1, column=1)
        
        # Кнопки управления
        ttk.Button(frame, text="Добавить точку", 
                command=self.add_calibration_point).grid(row=0, column=2, padx=5)
        ttk.Button(frame, text="Сбросить калибровку", 
                command=self.reset_calibration).grid(row=1, column=2, padx=5)
        
        # Список точек калибровки с полосой прокрутки
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=2, columnspan=3, sticky="ew", pady=5)
        
        # Создаем Listbox и Scrollbar
        self.calib_listbox = tk.Listbox(list_frame, height=4)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.calib_listbox.yview)
        self.calib_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.calib_listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Кнопка сохранения калибровки
        ttk.Button(frame, text="Сохранить калибровку", 
                command=self.save_calibration_data).grid(row=3, columnspan=3, pady=5)
        
        # Обновляем список
        self.update_calib_list()

    def create_last_measurements_panel(self):
        """Панель последних сохраненных измерений"""
        frame = ttk.LabelFrame(self.root, text="Последние измерения")
        frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Таблица измерений
        columns = ("timestamp", "distance", "hall_value", "thickness")
        self.measurements_tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # Настройка колонок
        self.measurements_tree.heading("timestamp", text="Время")
        self.measurements_tree.heading("distance", text="Расстояние, м")
        self.measurements_tree.heading("hall_value", text="Значение датчика")
        self.measurements_tree.heading("thickness", text="Толщина, мм")
        
        self.measurements_tree.column("timestamp", width=150)
        self.measurements_tree.column("distance", width=100)
        self.measurements_tree.column("hall_value", width=100)
        self.measurements_tree.column("thickness", width=100)
        
        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.measurements_tree.yview)
        self.measurements_tree.configure(yscrollcommand=scrollbar.set)
        
        self.measurements_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Кнопки управления
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Сохранить текущее", 
                  command=self.save_current_measurement).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Экспорт в CSV", 
                  command=self.export_measurements).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Очистить историю", 
                  command=self.clear_measurements).pack(side="right", padx=5)

    def update_calib_list(self):
        """Обновление списка калибровки"""
        self.calib_listbox.delete(0, tk.END)
        for h, t in self.calibration_points:
            self.calib_listbox.insert(tk.END, f"{h:>7.1f}  →  {t:.2f} мм")

    def add_calibration_point(self):
        """Добавление точки калибровки с проверкой"""
        try:
            hall_val = self.calib_hall_value.get()
            thickness = self.calib_thickness.get()
            
            # Проверка диапазона
            if not (self.hall_min_range.get() <= hall_val <= self.hall_max_range.get()):
                raise ValueError("Значение датчика вне допустимого диапазона")
            
            if thickness <= 0:
                raise ValueError("Толщина должна быть положительной")
            
            # Добавляем или обновляем точку
            existing_idx = next((i for i, (h, _) in enumerate(self.calibration_points) 
                              if abs(h - hall_val) < 1.0), None)
            
            if existing_idx is not None:
                self.calibration_points[existing_idx] = (hall_val, thickness)
            else:
                self.calibration_points.append((hall_val, thickness))
            
            # Сортируем по значению датчика
            self.calibration_points.sort()
            self.update_calib_list()
            
            # Пересчитываем все толщины
            self.update_thickness_calculation()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Невозможно добавить точку: {str(e)}")

    def reset_calibration(self):
        """Сброс калибровки"""
        self.calibration_points = []
        self.update_calibration_list()
        self.save_calibration_data()
        messagebox.showinfo("Информация", "Калибровка сброшена")

    def update_calibration_list(self):
        """Обновление списка точек калибровки"""
        self.calibration_listbox.delete(0, tk.END)
        for point in sorted(self.calibration_points, key=lambda x: x[0]):
            self.calibration_listbox.insert(tk.END, f"{point[0]:.1f} → {point[1]:.2f} мм")

    def update_thickness_calculation(self):
        """Полный пересчет толщин для всех данных"""
        if not hasattr(self, 'scan_data'):
            return
            
        # Пересчитываем все значения
        self.scan_data["thickness"] = [
            self.calculate_thickness(val) 
            for val in self.scan_data["front_sensor"]
        ]
        
        # Обновляем графики
        if hasattr(self, 'current_plot'):
            if self.current_plot == "thickness":
                self.plot_thickness()
            elif self.current_plot == "3d":
                self.visualize_data()

    def calculate_thickness(self, hall_value, temperature=None):
        """Расчет толщины с учетом калибровки"""
        try:
            # Если есть точки калибровки, используем их
            if self.calibration_points and len(self.calibration_points) >= 2:
                # Сортируем точки по значению датчика
                sorted_points = sorted(self.calibration_points, key=lambda x: x[0])
                x = [p[0] for p in sorted_points]
                y = [p[1] for p in sorted_points]
                
                # Линейная интерполяция между точками
                thickness = np.interp(hall_value, x, y)
            else:
                # Стандартный расчет, если калибровки нет
                temp_coef = 0.003
                normalized_hall = hall_value / 670  # Нормализация к [-1, 1]
                
                if temperature is not None:
                    temp_factor = 1 + temp_coef * (temperature - 20)
                else:
                    temp_factor = 1
                
                thickness = self.base_thickness.get() - (self.material_coef.get() * normalized_hall * temp_factor)
            
            return max(thickness, 0.1)  # Минимальная толщина 0.1 мм
            
        except Exception as e:
            self.logger.error(f"Ошибка расчета толщины: {str(e)}")
            return 0.1

    def save_current_measurement(self):
        """Сохранение текущего измерения"""
        try:
            if not self.scan_data or not self.scan_data.get("distances"):
                messagebox.showerror("Ошибка", "Нет данных для сохранения")
                return
            
            # Берем последние значения
            last_idx = -1
            measurement = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "distance": self.scan_data["distances"][last_idx],
                "hall_value": self.scan_data["front_sensor"][last_idx],
                "thickness": self.scan_data["thickness"][last_idx] if "thickness" in self.scan_data else 0
            }
            
            # Добавляем в начало списка
            self.last_measurements.insert(0, measurement)
            
            # Ограничиваем количество сохраненных измерений
            if len(self.last_measurements) > self.max_measurements:
                self.last_measurements = self.last_measurements[:self.max_measurements]
            
            # Обновляем таблицу
            self.update_measurements_table()
            
            # Сохраняем в файл
            self.save_measurements_data()
            
            messagebox.showinfo("Успех", "Измерение сохранено")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")

    def update_measurements_table(self):
        """Обновление таблицы последних измерений"""
        self.measurements_tree.delete(*self.measurements_tree.get_children())
        
        for meas in self.last_measurements:
            self.measurements_tree.insert("", "end", values=(
                meas["timestamp"],
                f"{meas['distance']:.3f}",
                f"{meas['hall_value']:.1f}",
                f"{meas['thickness']:.2f}"
            ))

    def clear_measurements(self):
        """Очистка истории измерений"""
        self.last_measurements = []
        self.update_measurements_table()
        self.save_measurements_data()
        messagebox.showinfo("Информация", "История измерений очищена")

    def export_measurements(self):
        """Экспорт измерений в CSV"""
        try:
            if not self.last_measurements:
                messagebox.showerror("Ошибка", "Нет данных для экспорта")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Сохранить измерения"
            )
            
            if not filename:
                return
            
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Заголовки
                writer.writerow(["Время", "Расстояние (м)", "Значение датчика", "Толщина (мм)"])
                
                # Данные
                for meas in self.last_measurements:
                    writer.writerow([
                        meas["timestamp"],
                        f"{meas['distance']:.3f}",
                        f"{meas['hall_value']:.1f}",
                        f"{meas['thickness']:.2f}"
                    ])
            
            messagebox.showinfo("Успех", f"Данные экспортированы в {filename}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")

    def load_saved_data(self):
        """Загрузка сохраненных данных при запуске"""
        try:
            # Загрузка калибровки
            try:
                with open("calibration.json", "r") as f:
                    self.calibration_points = json.load(f)
            except FileNotFoundError:
                self.calibration_points = []
            
            # Загрузка измерений
            try:
                with open("measurements.json", "r") as f:
                    self.last_measurements = json.load(f)
            except FileNotFoundError:
                self.last_measurements = []
                
        except Exception as e:
            self.logger.error(f"Ошибка загрузки данных: {str(e)}")

    def save_calibration_data(self):
        """Сохранение данных калибровки"""
        try:
            with open("calibration.json", "w") as f:
                json.dump(self.calibration_points, f)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения калибровки: {str(e)}")

    def save_measurements_data(self):
        """Сохранение истории измерений"""
        try:
            with open("measurements.json", "w") as f:
                json.dump(self.last_measurements, f)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения измерений: {str(e)}")

    def add_register(self):
        """Окно добавления нового регистра"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Добавить новый регистр")

        ttk.Label(dialog, text="Название:").grid(row=0, column=0)
        name_entry = ttk.Entry(dialog)
        name_entry.grid(row=0, column=1)

        ttk.Label(dialog, text="Адрес:").grid(row=1, column=0)
        addr_entry = ttk.Entry(dialog)
        addr_entry.grid(row=1, column=1)

        ttk.Label(dialog, text="Тип:").grid(row=2, column=0)
        type_combo = ttk.Combobox(dialog, values=["float", "int"])
        type_combo.grid(row=2, column=1)
        type_combo.current(0)

        def save_register():
            name = name_entry.get()
            addr = addr_entry.get()
            dtype = type_combo.get()

            if not name or not addr:
                messagebox.showerror("Ошибка", "Необходимо заполнить название и адрес!")
                return

            try:
                addr_int = int(addr)
                new_reg = Register(name, addr_int, dtype)
                self.registers.append(new_reg)
                self.update_registers_list()
                dialog.destroy()
            except ValueError:
                messagebox.showerror("Ошибка", "Адрес должен быть целым числом.")

        ttk.Button(dialog, text="Сохранить", command=save_register).grid(row=3, columnspan=2)

    def remove_register(self):
        """Удаляет выбранный регистр"""
        selected = self.registers_tree.selection()
        if not selected:
            return

        item = selected[0]
        values = self.registers_tree.item(item, 'values')
        reg_name = values[0]

        # Находим и удаляем регистр
        for i, reg in enumerate(self.registers):
            if reg.name == reg_name:
                del self.registers[i]
                break

        self.update_registers_list()

    def update_registers_list(self):
        """Обновляет список регистров в дереве TreeView"""
        self.registers_tree.delete(*self.registers_tree.get_children())

        for reg in self.registers:
            current_value = self.current_values.get(reg.name, "N/A")
            if isinstance(current_value, float):
                value_str = f"{current_value:.2f}"
            else:
                value_str = str(current_value)

            self.registers_tree.insert('', 'end', 
                                     values=(reg.name, reg.address, reg.data_type, value_str))

    def test_connection(self):
        """Тест подключения с проверкой чтения регистров"""
        try:
            with ModbusTcpClient(
                host=self.modbus_ip.get(),
                port=self.modbus_port.get(),
                timeout=2
            ) as client:
                if not client.connect():
                    messagebox.showerror("Ошибка", "Не удалось подключиться к устройству")
                    return False

                # Пробуем прочитать первый регистр
                try:
                    rr = client.read_holding_registers(0, 1, unit=self.slave_id.get())
                    if rr.isError():
                        messagebox.showwarning("Предупреждение", 
                                            "Устройство доступно, но чтение регистров не удалось")
                    else:
                        messagebox.showinfo("Успех", "Устройство доступно и отвечает!")
                    return True
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка чтения регистра: {str(e)}")
                    return False
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка подключения: {str(e)}")
            return False

    def read_modbus_data(self, client):
        """Чтение данных из всех активных регистров Modbus"""
        data = {}

        for reg in self.registers:
            if not reg.enabled.get():
                continue

            try:
                if reg.data_type == "float":
                    # Для float читаем 2 регистра (4 байта)
                    resp = client.read_holding_registers(
                        address=reg.address,
                        count=2,
                        slave=self.slave_id.get()
                    )

                    if resp.isError():
                        self.logger.warning(f"Ошибка Modbus при чтении {reg.name} (адрес {reg.address}): {resp}")
                        continue

                    # Проверяем, что получили 2 регистра
                    if len(resp.registers) != 2:
                        self.logger.warning(f"Неверное количество регистров для float: {len(resp.registers)}")
                        continue

                    decoder = BinaryPayloadDecoder.fromRegisters(
                        resp.registers,
                        byteorder=Endian.BIG,
                        wordorder=Endian.BIG
                    )
                    value = decoder.decode_32bit_float()

                elif reg.data_type == "int":
                    # Для int читаем 1 регистр (2 байта)
                    resp = client.read_holding_registers(
                        address=reg.address,
                        count=1,
                        slave=self.slave_id.get()
                    )

                    if resp.isError():
                        self.logger.warning(f"Ошибка Modbus при чтении {reg.name} (адрес {reg.address}): {resp}")
                        continue

                    if not resp.registers:
                        self.logger.warning(f"Пустой ответ для регистра {reg.name}")
                        continue

                    value = resp.registers[0]
                    # Преобразуем 16-битное целое со знаком, если значение > 32767
                    if value > 32767:
                        value -= 65536
                else:
                    continue

                data[reg.name] = value
                self.logger.debug(f"Успешно прочитан регистр {reg.name} (адрес {reg.address}): {value}")

            except Exception as e:
                self.logger.error(f"Ошибка чтения регистра {reg.name}: {str(e)}\n{traceback.format_exc()}")
                data[reg.name] = None

        return data

    def read_single_register(self, reg):
        """Читает один конкретный регистр Modbus"""
        if not self.client or not self.client.connected:
            if not self.connect_to_modbus():
                return None

        try:
            if reg.data_type == "float":
                resp = self.client.read_holding_registers(reg.address, 2, unit=self.slave_id.get())
                if resp.isError():
                    raise RuntimeError(f"Ошибка чтения регистра {reg.name}")

                decoder = BinaryPayloadDecoder.fromRegisters(
                    resp.registers, 
                    byteorder=Endian.BIG, 
                    wordorder=Endian.BIG
                )
                value = decoder.decode_32bit_float()
            elif reg.data_type == "int":
                resp = self.client.read_holding_registers(reg.address, 1, unit=self.slave_id.get())
                if resp.isError():
                    raise RuntimeError(f"Ошибка чтения регистра {reg.name}")

                value = resp.registers[0]
                # Преобразуем 16-битное целое со знаком, если значение > 32767
                if value > 32767:
                    value -= 65536
            else:
                return None

            # Обновляем интерфейс
            self.update_current_values({reg.name: value})
            return value

        except Exception as e:
            self.logger.error(f"Ошибка чтения регистра {reg.name}: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка чтения регистра {reg.name}: {str(e)}")
            return None

    def update_current_values(self, data):
        """Обновляет интерфейс и историю новыми значениями"""
        current_time = datetime.now()

        for name, value in data.items():
            if name in self.values_labels:
                # Обновление отображения
                formatted_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                self.values_labels[name].set(formatted_value)
                self.current_values[name] = value

                # Обновление истории
                with self.history_lock:
                    if name not in self.history:
                        self.history[name] = {'values': deque(maxlen=100), 'timestamps': deque(maxlen=100)}

                    self.history[name]['values'].append(value)
                    self.history[name]['timestamps'].append(current_time)

        # Обновляем список регистров
        self.update_registers_list()

    def connect_to_modbus(self):
        """Подключаемся к устройству Modbus"""
        try:
            self.client = ModbusTcpClient(
                host=self.modbus_ip.get(), 
                port=self.modbus_port.get(), 
                timeout=2
            )

            if self.client.connect():
                return True
            else:
                messagebox.showerror("Ошибка", "Не удалось подключиться к устройству Modbus")
                return False

        except Exception as e:
            self.logger.error(f"Ошибка подключения: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка подключения: {str(e)}")
            return False

    def start_auto_update(self):
        """Запускает поток автоматического обновления данных"""
        self.stop_event.clear()
        Thread(target=self.auto_update_thread, daemon=True).start()
        messagebox.showinfo("Автообновление", "Автоматическое обновление данных запущено")

    def stop_auto_update(self):
        """Останавливает поток автоматического обновления"""
        self.stop_event.set()
        messagebox.showinfo("Автообновление", "Автоматическое обновление данных остановлено")

    def auto_update_thread(self):
        """Автономный поток для постоянного обновления данных"""
        while not self.stop_event.is_set():
            try:
                if not self.client or not self.client.connected:
                    if not self.connect_to_modbus():
                        time.sleep(5)
                        continue

                self.read_all_registers()
                time.sleep(self.auto_update_interval)

            except Exception as e:
                self.logger.error(f"Ошибка автообновления: {str(e)}\n{traceback.format_exc()}")
                time.sleep(5)

    def read_all_registers(self):
        """Ручное чтение всех регистров Modbus"""
        if not self.client or not self.client.connected:
            if not self.connect_to_modbus():
                return

        try:
            data = self.read_modbus_data(self.client)
            if data:
                self.update_current_values(data)

        except Exception as e:
            self.logger.error(f"Ошибка чтения регистров: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("Ошибка", f"Ошибка чтения регистров: {str(e)}")

    def mainloop(self):
        """Основной цикл Tkinter-приложения"""
        try:
            self.root.mainloop()
        finally:
            # Гарантируем закрытие соединения при выходе
            if self.client and self.client.connected:
                self.client.close()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        app = MFLScannerApp(root)
        app.mainloop()
    except Exception as e:
        logging.error(f"Критическая ошибка: {str(e)}\n{traceback.format_exc()}")
        messagebox.showerror("Критическая ошибка", f"Произошла критическая ошибка: {str(e)}")