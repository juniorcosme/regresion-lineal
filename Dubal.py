import tkinter as tk 
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
import csv
import math
from collections import Counter
import numpy as np



def calcular_regresion(valores_x, valores_y): 
    total = len(valores_x)              
    promedio_x = sum(valores_x) / total
    promedio_y = sum(valores_y) / total
    numerador = sum((valores_x[i] - promedio_x) * (valores_y[i] - promedio_y) for i in range(total))
    denominador = sum((valores_x[i] - promedio_x) ** 2 for i in range(total))
    pendiente = numerador / denominador
    intercepto = promedio_y - pendiente * promedio_x
    return pendiente, intercepto

def calcular_error_mse(datos_x, datos_y, pendiente, intercepto): #Funcion para calcular el error cuadratico
    return sum((datos_y[i] - (pendiente * datos_x[i] + intercepto)) ** 2 for i in range(len(datos_x))) / len(datos_x)

def predecir_valor(x_input, pendiente, intercepto): #Funcion para hacer una prediccion
    return pendiente * x_input + intercepto

def distancia_euclidiana(punto_a, punto_b): 
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(punto_a, punto_b)))

def knn_clasificar(datos_entrenamiento, etiquetas, punto_nuevo, vecinos):  
    lista_distancias = [(distancia_euclidiana(punto_nuevo, datos_entrenamiento[i]), etiquetas[i]) #Calcula la distancia del nuevo punto a todos los puntos del dataset
  
                        for i in range(len(datos_entrenamiento))]
    lista_distancias.sort(key=lambda x: x[0])
    vecinos_cercanos = lista_distancias[:vecinos]
    clases = [etiqueta for _, etiqueta in vecinos_cercanos]
    return Counter(clases).most_common(1)[0][0]

#consola grafica para cargar los archivos CSV

class AplicacionIA:
    def __init__(self, ventana):
        self.ventana = ventana
        ventana.title("Proyecto IA - Byron Duval") 
        ventana.geometry("850x850")

        self.datos_x = []
        self.datos_y = []
        self.knn_x = []
        self.knn_y = []
        self.dimension_knn = 0

        pestañas = ttk.Notebook(ventana)
        self.pestaña_lr = ttk.Frame(pestañas)
        self.pestaña_knn = ttk.Frame(pestañas)
        pestañas.add(self.pestaña_lr, text="Regresión Lineal") 
        pestañas.add(self.pestaña_knn, text="KNN") 
        pestañas.pack(expand=1, fill="both")

 
        ttk.Button(self.pestaña_lr, text="Regresion CSV", command=self.cargar_csv_regresion).pack(pady=5)
        self.campo_x = ttk.Entry(self.pestaña_lr)
        self.campo_x.pack()
        self.campo_x.insert(0, "Ingrese valor X")
        ttk.Button(self.pestaña_lr, text="Calcular", command=self.ejecutar_regresion).pack(pady=5)
        self.label_resultado_lr = ttk.Label(self.pestaña_lr)
        self.label_resultado_lr.pack()
        self.figura_lr, self.grafico_lr = plt.subplots(figsize=(8, 6))
        self.canvas_lr = FigureCanvasTkAgg(self.figura_lr, master=self.pestaña_lr)
        self.canvas_lr.get_tk_widget().pack()

        
        ttk.Button(self.pestaña_knn, text="KNN CSV", command=self.cargar_csv_knn).pack(pady=5)
        ttk.Label(self.pestaña_knn, text="Ingrese valor K").pack()
        self.campo_k = ttk.Entry(self.pestaña_knn, width=5)
        self.campo_k.pack()
        self.campo_k.insert(0, "3")
        self.frame_inputs = ttk.Frame(self.pestaña_knn)
        self.frame_inputs.pack(pady=5)
        ttk.Button(self.pestaña_knn, text="Predecir", command=self.ejecutar_knn).pack(pady=5)
        self.label_knn = ttk.Label(self.pestaña_knn, font=("Arial", 14))
        self.label_knn.pack(pady=5)
        self.figura_knn, self.grafico_knn = plt.subplots(figsize=(8, 6))
        self.canvas_knn = FigureCanvasTkAgg(self.figura_knn, master=self.pestaña_knn)
        self.canvas_knn.get_tk_widget().pack()

# Funcion para detectar el delimitador de los archivos CSV bien sea , o ;
    def detectar_delimitador(self, ruta):
        with open(ruta, 'r') as archivo:
            muestra = archivo.read(1024)
            try:
                dialecto = csv.Sniffer().sniff(muestra, delimiters=";,")
                return dialecto.delimiter
            except:
                return ','

#Funcion para carga los arvhivos de regresion lineal 
    def cargar_csv_regresion(self):
        ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        separador = self.detectar_delimitador(ruta)
        self.datos_x = []
        self.datos_y = []
        with open(ruta) as archivo:
            lector = csv.reader(archivo, delimiter=separador)
            next(lector, None)
            for fila in lector:
                try:
                    x = float(fila[0])
                    y = float(fila[1])
                    self.datos_x.append(x)
                    self.datos_y.append(y)
                except:
                    pass
        messagebox.showinfo("Numero de datos cargados ", f"{len(self.datos_x)}")

#Funcion para cargar los datos KNN
    def cargar_csv_knn(self):
        ruta = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not ruta:
            return
        separador = self.detectar_delimitador(ruta)
        self.knn_x = []
        self.knn_y = []
        with open(ruta) as archivo:
            lector = csv.reader(archivo, delimiter=separador)
            next(lector, None)
            primera_fila = next(lector, None)
            if primera_fila is None:
                messagebox.showerror("Error", "Archivo sin datos")
                return
            self.dimension_knn = len(primera_fila) - 1
            self.crear_inputs(self.dimension_knn)
            try:
                caracteristicas = [float(v) for v in primera_fila[:-1]]
                etiqueta = primera_fila[-1]
                self.knn_x.append(caracteristicas)
                self.knn_y.append(etiqueta)
            except:
                pass
            for fila in lector:
                try:
                    caracteristicas = [float(v) for v in fila[:-1]]
                    etiqueta = fila[-1]
                    self.knn_x.append(caracteristicas)
                    self.knn_y.append(etiqueta)
                except:
                    pass
        messagebox.showinfo("Datos cargados", f"Dataset con {self.dimension_knn} dimensiones")

#Crear los puntos para crear las predicciones
    def crear_inputs(self, dimension):
        for widget in self.frame_inputs.winfo_children():
            widget.destroy()
        self.inputs = []
        ttk.Label(self.frame_inputs, text="Ingrese los puntos a clasificar").pack()
        for i in range(dimension):
            fila = ttk.Frame(self.frame_inputs)
            fila.pack(pady=2, anchor='w')
            ttk.Label(fila, text=f"Dimensión {i+1}").pack(side='left')
            entrada = ttk.Entry(fila, width=10)
            entrada.pack(side='left')
            self.inputs.append(entrada)

#Funcion para ejecutar la regresion lineal
    def ejecutar_regresion(self):
        if not self.datos_x:
            messagebox.showwarning("Error", "Cargue el CSV")
            return
        try:
            valor_x = float(self.campo_x.get())
        except:
            messagebox.showerror("Error", "Número invalido")
            return
        m, b = calcular_regresion(self.datos_x, self.datos_y)
        error = calcular_error_mse(self.datos_x, self.datos_y, m, b)
        prediccion = predecir_valor(valor_x, m, b)
        self.label_resultado_lr.config(
            text=f"y = {m:.2f}x + {b:.2f}\nMSE = {error:.3f}\nPredicción = {prediccion:.2f}"
        )
        self.grafico_lr.clear()
        self.grafico_lr.scatter(self.datos_x, self.datos_y, color="blue")
        linea_x = np.linspace(min(self.datos_x), max(self.datos_x), 100)
        self.grafico_lr.plot(linea_x, m * linea_x + b, color="red")
        self.grafico_lr.grid(True)
        self.canvas_lr.draw()

#Funcion para ejecutar el KNN
    def ejecutar_knn(self):
        if not self.knn_x:
            messagebox.showwarning("Error", "Cargar datos")
            return
        try:
            k = int(self.campo_k.get())
            if k <= 0 or k > len(self.knn_x):
                messagebox.showerror("Error", "Valor K inválido")
                return
        except:
            messagebox.showerror("Error", "K debe ser un numero entero")
            return
        try:
            punto = [float(e.get()) for e in self.inputs]
        except:
            messagebox.showerror("Error", "Ingrese valores numéricos")
            return
        if len(punto) != self.dimension_knn:
            messagebox.showerror("Error", f"Debe ingresar {self.dimension_knn} valores")
            return
        resultado = knn_clasificar(self.knn_x, self.knn_y, punto, k)
        self.label_knn.config(text=f"Clase predicha: {resultado}")

#grafica solo si los datos cargados tienen 2 dimensiones
        if self.dimension_knn != 2:
            self.grafico_knn.clear()
            self.grafico_knn.text(0.5, 0.5, "Visualización disponible para datos 2D",
                                  ha='center', va='center', fontsize=14)
            self.canvas_knn.draw()
            return

        self.grafico_knn.clear()
        X_array = np.array(self.knn_x)
        clases = list(set(self.knn_y))
        clase_a_num = {c: i for i, c in enumerate(clases)}
        y_numerico = np.array([clase_a_num[c] for c in self.knn_y])
        x_min, x_max = X_array[:, 0].min() - 1, X_array[:, 0].max() + 1
        y_min, y_max = X_array[:, 1].min() - 1, X_array[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        Z = []
        for xv, yv in zip(xx.ravel(), yy.ravel()):
            pred = knn_clasificar(self.knn_x, self.knn_y, [xv, yv], k)
            Z.append(clase_a_num[pred])
        Z = np.array(Z).reshape(xx.shape)
        cmap_fondo = ListedColormap(['#FFBBBB', '#BBBBFF', '#BBFFBB', '#FFFFBB'])
        cmap_puntos = ListedColormap(['red', 'blue', 'green', 'yellow'])
        self.grafico_knn.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_fondo)
        self.grafico_knn.scatter(X_array[:, 0], X_array[:, 1],
                                 c=y_numerico,
                                 cmap=cmap_puntos,
                                 edgecolor='black',
                                 s=80)
        self.grafico_knn.scatter(punto[0], punto[1],
                                 color='black',
                                 marker='X',
                                 s=150)
        self.grafico_knn.set_xlim(x_min, x_max)
        self.grafico_knn.set_ylim(y_min, y_max)
        self.grafico_knn.grid(True)
        self.canvas_knn.draw()

#inicio de todo el programa
ventana = tk.Tk() # para crear la ventana principal
app = AplicacionIA(ventana) #Inicia la aplicacion con todas la funciones creadas
ventana.mainloop() # permanece la ventana abierta

