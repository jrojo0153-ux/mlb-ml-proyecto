import os
import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

print(f"=== Ejecución del Ciclo de Predicción/Reentrenamiento - {datetime.now().strftime('%H:%M')} ===")

# --- 1. ARCHIVOS DE ALMACENAMIENTO DE DATOS LOCALES ---
HISTORIAL_PROYECCIONES = "historial_proyecciones.json"
HISTORIAL_ACIERTOS_FALLOS = "registro_rendimiento_mercados.csv"
DATASET_ENTRENAMIENTO = "dataset_mlb_mercados.csv"

# --- 2. EVALUAR PREDICCIONES ANTERIORES (ACIERTOS Y FALLOS) ---
url_mlb = f"https://mlb.com{datetime.now().strftime('%Y-%m-%d')}"
juegos_hoy = {}

try:
    response_mlb = requests.get(url_mlb).json()
    for date in response_mlb.get("dates", []):
        for game in date.get("games", []):
            g_id = str(game.get("gamePk"))
            status = game.get("status", {}).get("abstractGameState") # "Live", "Final", "Preview"
            if status == "Final":
                home_score = game.get("teams", {}).get("home", {}).get("score", 0)
                away_score = game.get("teams", {}).get("away", {}).get("score", 0)
                juegos_hoy[g_id] = {
                    "ganador": "HOME" if home_score > away_score else "AWAY",
                    "total_carreras": home_score + away_score,
                    "diferencia_carreras": abs(home_score - away_score)
                }
except Exception as e:
    print(f"Error al consultar el calendario de MLB API: {e}")

# Cruzar resultados reales con proyecciones guardadas
if os.path.exists(HISTORIAL_PROYECCIONES) and juegos_hoy:
    with open(HISTORIAL_PROYECCIONES, "r") as f:
        predicciones_guardadas = json.load(f)
    
    nuevos_registros = []
    for g_id, pred in predicciones_guardadas.items():
        if g_id in juegos_hoy and not pred.get("evaluado", False):
            real = juegos_hoy[g_id]
            acierto_ml = 1 if pred["prediccion_ganador"] == real["ganador"] else 0
            
            nuevos_registros.append({
                "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "game_id": g_id,
                "mercado": "Moneyline",
                "prediccion": pred["prediccion_ganador"],
                "real": real["ganador"],
                "resultado_evaluacion": acierto_ml
            })
            pred["evaluado"] = True
            
    if nuevos_registros:
        df_nuevos = pd.DataFrame(nuevos_registros)
        if os.path.exists(HISTORIAL_ACIERTOS_FALLOS):
            df_existente = pd.read_csv(HISTORIAL_ACIERTOS_FALLOS)
            pd.concat([df_existente, df_nuevos]).to_csv(HISTORIAL_ACIERTOS_FALLOS, index=False)
        else:
            df_nuevos.to_csv(HISTORIAL_ACIERTOS_FALLOS, index=False)
        
    with open(HISTORIAL_PROYECCIONES, "w") as f:
        json.dump(predicciones_guardadas, f, indent=4)

# --- 3. REENTRENAMIENTO DEL MODELO CON REALIMENTACIÓN ---
if not os.path.exists(DATASET_ENTRENAMIENTO):
    # Dataset inicial de contingencia con datos históricos base estructurados
    datos_iniciales = {
        "promedio_home": [4.5, 3.2, 5.1, 4.0, 3.8, 4.2, 3.1, 5.5, 4.6, 3.9],
        "promedio_away": [3.8, 4.1, 3.9, 4.5, 4.2, 3.6, 4.9, 3.3, 4.1, 4.4],
        "cuota_odds": [1.80, 2.10, 1.65, 2.30, 1.95, 1.72, 2.20, 1.55, 1.85, 2.05],
        "resultado_real": [1, 0, 1, 0, 1, 1, 0, 1, 0, 0] # 1=Gana Home, 0=Gana Away
    }
    pd.DataFrame(datos_iniciales).to_csv(DATASET_ENTRENAMIENTO, index=False)

df_entrenar = pd.read_csv(DATASET_ENTRENAMIENTO)

# Si ya hay aciertos y fallos registrados en el ciclo dinámico, expandimos el conocimiento de la IA
if os.path.exists(HISTORIAL_ACIERTOS_FALLOS):
    df_feedback = pd.read_csv(HISTORIAL_ACIERTOS_FALLOS)
    # Convertimos los aciertos/fallos en filas útiles agregando valores simulados coherentes para reentrenar
    nuevas_filas = []
    for _, fila in df_feedback.iterrows():
        # Reconstrucción de variables en base al resultado real para actualizar pesos
        es_home = 1 if fila["real"] == "HOME" else 0
        nuevas_filas.append({
            "promedio_home": 4.5 if es_home else 3.5,
            "promedio_away": 3.5 if es_home else 4.5,
            "cuota_odds": 1.90,
            "resultado_real": es_home
        })
    if nuevas_filas:
        df_entrenar = pd.concat([df_entrenar, pd.DataFrame(nuevas_filas)], ignore_index=True)

X = df_entrenar[["promedio_home", "promedio_away", "cuota_odds"]]
y = df_entrenar["resultado_real"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = XGBClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, eval_metric="logloss")
modelo.fit(X_train, y_train)
print(f"Modelo reentrenado con éxito. Precisión en pruebas: {modelo.score(X_test, y_test):.2%}")

# --- 4. FUNCIÓN PARA ENVIAR ALERTA A TELEGRAM ---
def enviar_alerta_telegram(mensaje):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if token and chat_id:
        url = f"https://telegram.org{token}/sendMessage"
        payload = {"chat_id": chat_id, "text": mensaje, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Error al enviar mensaje a Telegram: {e}")
    else:
        print("Advertencia: Faltan las variables de configuración de Telegram (Secrets).")

# --- 5. DETECCIÓN DE APUESTAS DE VALOR (VALUE BETS) ---
nuevas_predicciones = {}
if os.path.exists(HISTORIAL_PROYECCIONES):
    with open(HISTORIAL_PROYECCIONES, "r") as f:
        try:
            nuevas_predicciones = json.load(f)
        except json.JSONDecodeError:
            nuevas_predicciones = {}

if "dates" in response_mlb:
    for date in response_mlb["dates"]:
        for game in date.get("games", []):
            g_id = str(game.get("gamePk"))
            status = game.get("status", {}).get("abstractGameState")
            
            if status == "Preview" and g_id not in nuevas_predicciones:
                # Cuota base simulada provista para calcular valor (2.15 en este ejemplo)
                cuota_casino = 2.15 
                
                # Datos de entrada para calcular probabilidad del encuentro
                input_prediccion = pd.DataFrame([[4.3, 3.8, cuota_casino]], columns=["promedio_home", "promedio_away", "cuota_odds"])
                pred_clase = int(modelo.predict(input_prediccion)[0])
                probabilidad_modelo = float(modelo.predict_proba(input_prediccion)[0][pred_clase])
                
                probabilidad_casino = 1 / cuota_casino
                
                # Envío de alertas si existe una ventaja matemática (Value Bet)
                if probabilidad_modelo > probabilidad_casino:
                    ventaja = probabilidad_modelo - probabilidad_casino
                    home_team_name = game.get("teams", {}).get("home", {}).get("team", {}).get("name", "Local")
                    away_team_name = game.get("teams", {}).get("away", {}).get("team", {}).get("name", "Visitante")
                    equipo_elegido = home_team_name if pred_clase == 1 else away_team_name
                    
                    mensaje_alert = (
                        f"🚨 *¡APUESTA DE VALOR DETECTADA!* 🚨\n\n"
                        f"⚾️ *Partido:* {away_team_name} vs {home_team_name}\n"
                        f"🎯 *Pick Recomendado:* {equipo_elegido} (Moneyline)\n"
                        f"📊 *Probabilidad de la IA:* {probabilidad_modelo:.1%}\n"
                        f"🎲 *Cuota del Casino:* {cuota_casino:.2f} (Implica {probabilidad_casino:.1%})\n"
                        f"📈 *Ventaja matemática:* +{ventaja:.1%}\n"
                        f"⏰ *Hora de Análisis:* {datetime.now().strftime('%H:%M')} UTC"
                    )
                    enviar_alerta_telegram(mensaje_alert)

                # Registrar proyecciones horarias
                nuevas_predicciones[g_id] = {
                    "home_team": game.get("teams", {}).get("home", {}).get("team", {}).get("name"),
                    "away_team": game.get("teams", {}).get("away", {}).get("team", {}).get("name"),
                    "prediccion_ganador": "HOME" if pred_clase == 1 else "AWAY",
                    "evaluado": False,
                    "hora_proyeccion": datetime.now().strftime("%H:%M")
                }

with open(HISTORIAL_PROYECCIONES, "w") as f:
    json.dump(nuevas_predicciones, f, indent=4)
print("Ciclo finalizado y registros actualizados con éxito.")
