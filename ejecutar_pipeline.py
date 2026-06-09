import os
import json
import requests
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Formatos de fecha para los endpoints de ESPN
fecha_hoy_espn = datetime.now().strftime('%Y%m%d')
print(f"=== Ciclo Pro: ESPN + The Odds API (Mercado: 5-Innings) - {datetime.now().strftime('%H:%M')} ===")

# --- ARCHIVOS LOCALES ---
HISTORIAL_PROYECCIONES = "historial_proyecciones.json"
HISTORIAL_ACIERTOS_FALLOS = "registro_rendimiento_mercados.csv"
DATASET_ENTRENAMIENTO = "dataset_mlb_mercados.csv"

# --- 1. EXTRAER ESTADÍSTICAS Y MARCADORES EN VIVO DESDE ESPN ---
def obtener_datos_espn():
    """
    Se conecta al marcador de ESPN para verificar las carreras anotadas en las primeras 
    5 entradas o el resultado final si el juego ya concluyó.
    """
    url_scoreboard = f"https://espn.com{fecha_hoy_espn}"
    juegos_finalizados_5i = {}
    juegos_programados = []
    
    try:
        res = requests.get(url_scoreboard).json()
        for event in res.get("events", []):
            g_id = str(event.get("id"))
            status = event.get("status", {}).get("type", {}).get("state") # "pre", "in", "post"
            
            competidores = event.get("competitions", [{}]).get("competitors", [])
            home_box = next((c for c in competidores if c.get("homeAway") == "home"), {})
            away_box = next((c for c in competidores if c.get("homeAway") == "away"), {})
            
            home_name = home_box.get("team", {}).get("displayName")
            away_name = away_box.get("team", {}).get("displayName")
            
            # Verificar si ya pasaron las primeras 5 entradas (analizando las líneas por entrada de ESPN)
            # Si el juego ya está en progreso avanzado ("in") o finalizado ("post")
            if status in ["in", "post"]:
                # ESPN desglosa las carreras por entrada en 'linescores'
                home_linescore = home_box.get("linescores", [])
                away_linescore = away_box.get("linescores", [])
                
                # Necesitamos que se hayan jugado al menos 5 entradas
                if len(home_linescore) >= 5 and len(away_linescore) >= 5:
                    try:
                        runs_home_5i = sum(int(home_linescore[i].get("value", 0)) for i in range(5))
                        runs_away_5i = sum(int(away_linescore[i].get("value", 0)) for i in range(5))
                        
                        if runs_home_5i != runs_away_5i: # Ignoramos empates momentáneamente para Moneyline
                            juegos_finalizados_5i[g_id] = "HOME" if runs_home_5i > runs_away_5i else "AWAY"
                    except Exception:
                        pass
                        
            if status == "pre":
                juegos_programados.append({
                    "game_id": g_id,
                    "home_team": home_name,
                    "away_team": away_name
                })
    except Exception as e:
        print(f"Error consultando el marcador de ESPN: {e}")
        
    return juegos_finalizados_5i, juegos_programados

# Ejecutar consulta a ESPN
juegos_finalizados_5i, juegos_programados = obtener_datos_espn()

# --- 2. EXTRAER CUOTAS REALES EN VIVO PARA 5-INNINGS (THE ODDS API) ---
def obtener_cuotas_reales_5i():
    """Descarga cuotas vigentes de Moneyline específicas para las primeras 5 entradas (h2h_1st_5_innings)"""
    api_key = os.environ.get("ODDS_API_KEY")
    odds_map = {}
    if not api_key:
        print("Falta ODDS_API_KEY en las variables de entorno.")
        return odds_map
        
    # Usamos el mercado específico 'h2h_1st_5_innings' provisto por la API
    url = f"https://the-odds-api.com{api_key}&regions=us&markets=h2h_1st_5_innings&oddsFormat=decimal"
    try:
        res = requests.get(url).json()
        for match in res:
            home = match.get("home_team")
            away = match.get("away_team")
            bookmakers = match.get("bookmakers", [])
            if bookmakers:
                # Buscamos el mercado en el primer operador de apuestas que lo ofrezca
                markets = bookmakers[0].get("markets", [])
                market_5i = next((m for m in markets if m.get("key") == "h2h_1st_5_innings"), None)
                
                if market_5i:
                    outcomes = market_5i.get("outcomes", [])
                    cuota_home = next((o["price"] for o in outcomes if o["name"] == home), 1.90)
                    cuota_away = next((o["price"] for o in outcomes if o["name"] == away), 1.90)
                    
                    odds_map[home] = {"cuota": cuota_home, "rival": away, "rol": "HOME"}
                    odds_map[away] = {"cuota": cuota_away, "rival": home, "rol": "AWAY"}
    except Exception as e:
        print(f"Error procesando las cuotas de 5 Innings de The Odds API: {e}")
    return odds_map

cuotas_vivas = obtener_cuotas_reales_5i()

# --- 3. EVALUACIÓN CONTINUA DE LAS 5 ENTRADAS (FEEDBACK LOOP) ---
if os.path.exists(HISTORIAL_PROYECCIONES) and juegos_finalizados_5i:
    with open(HISTORIAL_PROYECCIONES, "r") as f:
        predicciones_guardadas = json.load(f)
    
    nuevos_logs = []
    for g_id, pred in predicciones_guardadas.items():
        if g_id in juegos_finalizados_5i and not pred.get("evaluado", False):
            ganador_real_5i = juegos_finalizados_5i[g_id]
            acierto = 1 if pred["prediccion_ganador"] == ganador_real_5i else 0
            
            nuevos_logs.append({
                "fecha_hora": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "game_id": g_id,
                "home_team": pred["home_team"],
                "away_team": pred["away_team"],
                "prediccion": pred["prediccion_ganador"],
                "real_5i": ganador_real_5i,
                "resultado_evaluacion": acierto
            })
            pred["evaluado"] = True
            
            # Registrar experiencia en el Dataset enfocado en 5 Innings
            if os.path.exists(DATASET_ENTRENAMIENTO):
                df_data = pd.read_csv(DATASET_ENTRENAMIENTO)
                nueva_linea = pd.DataFrame([{
                    "cuota_movimiento": float(pred.get("meta_cuota", 1.90)),
                    "resultado_real": 1 if ganador_real_5i == "HOME" else 0
                }])
                pd.concat([df_data, nueva_linea], ignore_index=True).to_csv(DATASET_ENTRENAMIENTO, index=False)

    if nuevos_logs:
        df_log = pd.DataFrame(nuevos_logs)
        df_log.to_csv(HISTORIAL_ACIERTOS_FALLOS, mode='a', header=not os.path.exists(HISTORIAL_ACIERTOS_FALLOS), index=False)
        
    with open(HISTORIAL_PROYECCIONES, "w") as f:
        json.dump(predicciones_guardadas, f, indent=4)

# --- 4. MODELO DE CLASIFICACIÓN MACHINE LEARNING ---
if not os.path.exists(DATASET_ENTRENAMIENTO):
    df_init = pd.DataFrame(columns=["cuota_movimiento", "resultado_real"])
    df_init.loc[0] = [1.85, 1]
    df_init.loc[1] = [2.20, 0]
    df_init.loc[2] = [1.95, 1]
    df_init.to_csv(DATASET_ENTRENAMIENTO, index=False)

df_training = pd.read_csv(DATASET_ENTRENAMIENTO)
X = df_training[["cuota_movimiento"]]
y = df_training["resultado_real"]

modelo = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, eval_metric="logloss")
modelo.fit(X, y)

# --- 5. DETECTAR VALOR Y NOTIFICAR A TELEGRAM ---
def enviar_telegram(msg):
    t_tok = os.environ.get("TELEGRAM_TOKEN")
    t_id = os.environ.get("TELEGRAM_CHAT_ID")
    if t_tok and t_id:
        url = f"https://telegram.org{t_tok}/sendMessage"
        requests.post(url, json={"chat_id": t_id, "text": msg, "parse_mode": "Markdown"})

proyecciones_actuales = {}
if os.path.exists(HISTORIAL_PROYECCIONES):
    with open(HISTORIAL_PROYECCIONES, "r") as f:
        try: proyecciones_actuales = json.load(f)
        except: proyecciones_actuales = {}

try:
    for juego in juegos_programados:
        g_id = juego["game_id"]
        
        # Filtro de duplicados
        if g_id in proyecciones_actuales:
            continue
            
        home_team = juego["home_team"]
        away_team = juego["away_team"]
        
        # Si no encontramos cuota específica de 5 innings para ese equipo, usamos una cuota base por defecto
        cuota_info = cuotas_vivas.get(home_team, {"cuota": 1.85})
        cuota_mercado = cuota_info["cuota"]
        
        input_live = pd.DataFrame([[cuota_mercado]], columns=["cuota_movimiento"])
        
        pred_clase = int(modelo.predict(input_live)[0])
        prob_ia = float(modelo.predict_proba(input_live)[0][pred_clase])
        prob_casino = 1 / cuota_mercado
        
        alerta_enviada = False
        # Filtro de ventaja matemática estricto (+3%)
        if prob_ia > (prob_casino + 0.03):
            ventaja = prob_ia - prob_casino
            pick = home_team if pred_clase == 1 else away_team
            
            mensaje = (
                f"🛡 *VALOR DETECTADA: 1AS 5 ENTRADAS (5I)* 🛡\n\n"
                f"🏟 *Partido:* {away_team} vs {home_team}\n"
                f"🎯 *Pick Recomendado:* {pick} (Línea de 5 Innings)\n"
                f"🎰 *Cuota Casino (5I):* {cuota_mercado:.2f}\n"
                f"📈 *Ventaja contra la Casa:* +{ventaja:.1%}\n"
                f"⏰ *ID de Juego ESPN:* {g_id}"
            )
            enviar_telegram(mensaje)
            alerta_enviada = True
        
        proyecciones_actuales[g_id] = {
            "home_team": home_team,
            "away_team": away_team,
            "prediccion_ganador": "HOME" if pred_clase == 1 else "AWAY",
            "meta_cuota": cuota_mercado,
            "alerta_emitida": alerta_enviada,
            "evaluado": False
        }
except Exception as e:
    print(f"Error en el bloque de procesamiento y envío: {e}")

with open(HISTORIAL_PROYECCIONES, "w") as f:
