
# Importar librerías necesarias
import os
import kagglehub
import pandas as pd
import argparse
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

# Función para obtener los argumentos de la línea de comandos
def obtener_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de XGBoost para predecir la potabilidad del agua"
    )

    parser.add_argument("--n_estimators", type=int, default=200,
                        help="Número de árboles (default: 200)")

    parser.add_argument("--max_depth", type=int, default=6,
                        help="Profundidad máxima (default: 6)")

    parser.add_argument("--learning_rate", type=float, default=0.05,
                        help="Tasa de aprendizaje (default: 0.05)")

    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria (default: 42)")

    return parser.parse_args()

# Función para cargar los datos desde Kaggle
def cargar_datos():
    path = kagglehub.dataset_download("adityakadiwal/water-potability")
    archivo = os.path.join(path, "water_potability.csv")
    return pd.read_csv(archivo)

# Función para preprocesar los datos
def preprocesar_datos(df, seed):
    X = df.drop(columns="Potability")
    y = df["Potability"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return X_train, X_test, y_train, y_test

# Función para entrenar el modelo
def entrenar_modelo(X_train, y_train, args):
    modelo = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=1.0,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=5,
        scale_pos_weight=1.6,
        eval_metric="logloss",
        random_state=args.seed
    )

    modelo.fit(X_train, y_train)
    return modelo

# Función para evaluar el modelo
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    for nombre, valor in metricas.items():
        print(f"{nombre}: {valor:.4f}")

    return metricas

# Función para guardar la matriz de confusión
def guardar_matriz_confusion(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    matriz = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=matriz)
    disp.plot(cmap="Blues")

    plt.savefig("matriz_confusion.png")
    plt.close()

    mlflow.log_artifact("matriz_confusion.png")

# Función principal
def main():
    args = obtener_argumentos()

    print(f"Configuración: {vars(args)}")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Water_Potability_XGBoost")

    with mlflow.start_run():
        # Registrar hiperparámetros
        for nombre, valor in vars(args).items():
            mlflow.log_param(nombre, valor)

        # Preparar datos y entrenar
        df = cargar_datos()
        X_train, X_test, y_train, y_test = preprocesar_datos(df, args.seed)
        modelo = entrenar_modelo(X_train, y_train, args)

        # Evaluar y registrar métricas
        metricas = evaluar_modelo(modelo, X_test, y_test)

        for nombre, valor in metricas.items():
            mlflow.log_metric(nombre, valor)

        # Registrar matriz de confusión
        guardar_matriz_confusion(modelo, X_test, y_test)

        # Registrar modelo entrenado
        mlflow.xgboost.log_model(modelo, name="modelo_xgboost")

# Punto de entrada del script
if __name__ == "__main__":
    main()
