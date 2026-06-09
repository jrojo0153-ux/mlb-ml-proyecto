import os
import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

fecha_hoy_espn = datetime.now().strftime('%Y%m%d')
print(f"=== Ciclo Pro: Totales MLB (Over/Under) - {datetime.now().strftime('%H:%M')} ===")

# --- ARCHIVOS LOCALES ---
HISTORIAL_PROYECCIONES = "historial_totales.json"
HISTORIAL_ACIERTOS_FALLOS = "registro_totales_mercados.csv"
DATASET_ENTRENAMIENTO = "dataset_mlb_totales.csv"

# --- 1. EXTRAER MARCADORES DE TOTALES DESDE ESPN ---
def obtener_datos_espn():
    url_scoreboard = f"https://espn.com{fecha_hoy_espn}"
    juegos_finalizados = {}
    juegos_programados = []
    
    try:
        res = requests.get(url_scoreboard).json()
        for event in res.get("events", []):
            g_id = str(event.get("id"))
            status = event.get("status", {}).get("type", {}).get("state")
            
            competidores = event.get("competitions", [{}]).get("competitors", [])
            home_name = next((c for c in competidores if c.get("homeAway") == "home"), {}).get("team", {}).get("displayName")
            away_name = next((c for c in competidores if c.get("homeAway") == "away"), {}).get("team", {}).get("displayName")
            
            if status == "post":
                home_score = int(next((c for c in competidores if c.get("homeAway") == "home"), {}).get("score", 0))
                away_score = int(next((c for c in competidores if c.get("homeAway") == "away"), {}).get("score", 0))
                # Guardamos la suma total de carreras reales del encuentro
                juegos_finalizados[g_id] = home_score + away_score
                
            elif status == "pre":
                juegos_programados.append({
                    "game_id": g_id,
                    "home_team": home_name,
                    "away_team": away_name
                })
    except Exception as e:
        print(f"Error en scoreboard de ESPN: {e}")
    return juegos_finalizados, juegos_programados

juegos_finalizados, juegos_programados = obtener_datos_espn()

# --- 2. EXTRAER CUOTAS DE OVER/UNDER EN VIVO (THE ODDS API) ---
def obtener_lineas_totales():
    api_key = os.environ.get("ODDS_API_KEY")
    totals_map = {}
    if not api_key:
        print("Falta ODDS_API_KEY en las variables de entorno.")
        return totals_map
        
    url = f"https://the-odds-api.com{api_key}&regions=us&markets=totals&oddsFormat=decimal"
    try:
        res = requests.get(url).json()
        for match in res:
            home = match.get("home_team")
            bookmakers = match.get("bookmakers", [])
            if bookmakers:
                markets = bookmakers[0].get("markets", [])
                market_totals = next((m for m in markets if m.get("key") == "totals"), None)
                
                if market_totals:
                    outcomes = market_totals.get("outcomes", [])
                    # Las líneas de Over y Under comparten el mismo punto numérico (ej. 8.5 carreras)
                    linea_puntos = float(outcomes[0].get("point", 8.5))
                    
                    cuota_over = next((o["price"] for o in outcomes if o["name"] == "Over"), 1.90)
                    cuota_under = next((o["price"] for o in outcomes if o["name"] == "Under"), 1.90)
                    
                    totals_map[home] = {
                        "linea": linea_puntos,
                        "cuota_over": cuota_over,
                        "cuota_under": cuota_under
                    }
    except Exception as e:
        print(f"Error procesando líneas de totales: {e}")
    return totals_map

cuotas_totales = obtener_lineas_totales()

# --- 3. EVALUACIÓN Y REENTRENAMIENTO (FEEDBACK LOOP DE TOTALES) ---
if os.path.exists(HISTORIAL_PROYECCIONES) and juegos_finalizados:
    with open(HISTORIAL_PROYECCIONES, "r") as f:
        predicciones_guardadas = json.load(f)
    
    nuevos_logs = []
    for g_id, pred in predicciones_guardadas.items():
        if g_id in juegos_finalizados and not pred.get("evaluado", False):
            total_real_carreras = juegos_finalizados[g_id]
            linea_proyectada = float(pred["linea_casino"])
            
            # Verificar si se cumplió el Over o el Under
            resultado_real_string = "OVER" if total_real_carreras > linea_proyectada else "UNDER"
            if total_real_carreras == linea_proyectada: resultado_real_string = "PUSH" # Empate con la casa
            
            acierto = 1 if pred["prediccion_tipo"] == resultado_real_string else 0
            
            nuevos_logs.append({
                "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "game_id": g_id,
                "linea_casino": linea_proyectada,
                "total_real": total_real_carreras,
                "prediccion": pred["prediccion_tipo"],
                "resultado_evaluacion": acierto
            })
            pred["evaluado"] = True
            
            if os.path.exists(DATASET_ENTRENAMIENTO):
                df_data = pd.read_csv(DATASET_ENTRENAMIENTO)
                nueva_linea = pd.DataFrame([{
                    "linea_puntos": linea_proyectada,
                    "cuota_movimiento": float(pred.get("meta_cuota", 1.90)),
                    "resultado_binario": 1 if resultado_real_string == "OVER" else 0
                }])
                pd.concat([df_data, nueva_linea], ignore_index=True).to_csv(DATASET_ENTRENAMIENTO, index=False)

    if nuevos_logs:
        pd.DataFrame(nuevos_logs).to_csv(HISTORIAL_ACIERTOS_FALLOS, mode='a', header=not os.path.exists(HISTORIAL_ACIERTOS_FALLOS), index=False)
    with open(HISTORIAL_PROYECCIONES, "w") as f:
        json.dump(predicciones_guardadas, f, indent=4)

# --- 4. ENTRENAMIENTO DEL MODELO XGBOOST PARA TOTALES ---
if not os.path.exists(DATASET_ENTRENAMIENTO):
    df_init = pd.DataFrame(columns=["linea_puntos", "cuota_movimiento", "resultado_binario"])
    df_init.loc = [8.5, 1.90, 1]
    df_init.loc = [7.5, 1.85, 0]
    df_init.loc = [9.0, 1.95, 1]
    df_init.to_csv(DATASET_ENTRENAMIENTO, index=False)

df_training = pd.read_csv(DATASET_ENTRENAMIENTO)
X = df_training[["linea_puntos", "cuota_movimiento"]]
y = df_training["resultado_binario"]

modelo = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, eval_metric="logloss")
modelo.fit(X, y)

# --- 5. DETECTAR JUEGOS QUE CUMPLEN EN TOTALES Y ENVIAR ALERTA ---
def enviar_telegram(msg):
    t_tok = os.environ.get("TELEGRAM_TOKEN")
    t_id = os.environ.get("TELEGRAM_CHAT_ID")
    if t_tok and t_id:
        requests.post(f"https://telegram.org{t_tok}/sendMessage", json={"chat_id": t_id, "text": msg, "parse_mode": "Markdown"})

proyecciones_actuales = {}
if os.path.exists(HISTORIAL_PROYECCIONES):
    with open(HISTORIAL_PROYECCIONES, "r") as f:
        try: proyecciones_actuales = json.load(f)
        except: proyecciones_actuales = {}

for juego in juegos_programados:
    g_id = juego["game_id"]
    if g_id in proyecciones_actuales:
        continue # Filtro de duplicación activado
        
    home_team = juego["home_team"]
    away_team = juego["away_team"]
    
    # Obtener línea de carreras real provista por el casino
    linea_info = cuotas_totales.get(home_team, {"linea": 8.5, "cuota_over": 1.90, "cuota_under": 1.90})
    linea_actual = linea_info["linea"]
    
    # Evaluar probabilidad de Altas (Over)
    input_over = pd.DataFrame([[linea_actual, linea_info["cuota_over"]]], columns=["linea_puntos", "cuota_movimiento"])
    prob_over = float(modelo.predict_proba(input_over)[1])
    prob_casino_over = 1 / linea_info["cuota_over"]
    
    # Evaluar probabilidad de Bajas (Under)
    prob_under = 1 - prob_over
    prob_casino_under = 1 / linea_info["cuota_under"]
    
    prediccion_tipo = "OVER" if prob_over > prob_under else "UNDER"
    prob_seleccionada = prob_over if prediccion_tipo == "OVER" else prob_under
    prob_casino_seleccionada = prob_casino_over if prediccion_tipo == "OVER" else prob_casino_under
    cuota_seleccionada = linea_info["cuota_over"] if prediccion_tipo == "OVER" else linea_info["cuota_under"]
    
    # El juego "CUMPLE" si la ventaja contra la casa es mayor al 3%
    if prob_seleccionada > (prob_casino_seleccionada + 0.03):
        ventaja = prob_seleccionada - prob_casino_seleccionada
        
        mensaje = (
            f"📈 *ALERTA DE TOTALES DETECTADA (O/U)* 📈\n\n"
            f"🏟 *Partido:* {away_team} vs {home_team}\n"
            f"📊 *Línea de Carreras:* {linea_actual} goles/carreras\n"
            f"🎯 *Pick:* {prediccion_tipo} (Altas/Bajas)\n"
            f"🎰 *Cuota Real:* {cuota_seleccionada:.2f}\n"
            f"📈 *Ventaja contra la Casa:* +{ventaja:.1%}\n"
            f"⏰ *ID de Juego ESPN:* {g_id}"
        )
        enviar_telegram(mensaje)
        
    proyecciones_actuales[g_id] = {
        "home_team": home_team,
        "away_team": away_team,
        "linea_casino": linea_actual,
        "prediccion_tipo": prediccion_tipo,
        "meta_cuota": cuota_seleccionada,
        "evaluado": False
    }

with open(HISTORIAL_PROYECCIONES, "w") as f:
    json.dump(proyecciones_actuales, f, indent=4)
print("Análisis de totales de la jornada concluido con éxito.")
