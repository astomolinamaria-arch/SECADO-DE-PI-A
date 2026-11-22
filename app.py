import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Configuración de Marca ----------
st.set_page_config(
    page_title="ThermoFruit Pro | Optimización de Deshidratado Industrial",
    page_icon="🍍",
    layout="wide",
)

# Estilo CSS limpio para tarjetas y encabezados
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem;}
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 12px 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Base de datos de recetas por fruta ----------
RECETAS_FRUTA = {
    "Piña Golden (Rodajas 6 mm)": {"M0": 3.00, "Me": 0.10, "M_obj": 0.20, "a": 0.045, "b": 3.0, "T_max_cal": 65.0, "rendimiento_pulpa": 0.55},
    "Mango Kent (Láminas 5 mm)": {"M0": 4.20, "Me": 0.12, "M_obj": 0.18, "a": 0.042, "b": 2.8, "T_max_cal": 62.0, "rendimiento_pulpa": 0.60},
    "Banano (Monedas 4 mm)": {"M0": 2.60, "Me": 0.09, "M_obj": 0.15, "a": 0.048, "b": 3.1, "T_max_cal": 68.0, "rendimiento_pulpa": 0.65},
}

# ---------- Barra Lateral: Configuración de Planta ----------
st.sidebar.title("🍍 ThermoFruit Pro")
st.sidebar.caption("Gemelo Digital para Plantas Deshidratadoras")
st.sidebar.divider()

fruta_sel = st.sidebar.selectbox("Producto a procesar", list(RECETAS_FRUTA.keys()))
params = RECETAS_FRUTA[fruta_sel]

st.sidebar.subheader("1. Capacidad y Costos de Planta")
kg_fresca_lote = st.sidebar.number_input("Carga de fruta fresca por lote (kg)", 20.0, 2000.0, 150.0, 10.0)
costo_fruta_kg = st.sidebar.number_input("Costo fruta fresca (S/ por kg)", 0.50, 20.00, 2.20, 0.10)
precio_venta_kg = st.sidebar.number_input("Precio venta deshidratado (S/ por kg)", 10.0, 150.0, 48.0, 1.0)
lotes_mes = st.sidebar.number_input("Lotes proyectados al mes", 1, 2000, 520, 10)

st.sidebar.subheader("2. Energía y Control Térmico")
P_base = st.sidebar.number_input("Potencia nominal del horno (kW)", 1.0, 150.0, 12.0, 0.5)
tarifa = st.sidebar.number_input("Tarifa eléctrica industrial (S/ por kWh)", 0.10, 5.00, 0.75, 0.05)
T_calidad = st.sidebar.slider("Límite térmico de calidad (°C)", 55.0, 75.0, params["T_max_cal"], 0.5)
Ta_manual = st.sidebar.slider("Temperatura de operación actual (°C)", 55.0, 75.0, 60.0, 1.0)

# ---------- Motor de Simulación (RK4) ----------
M0, Me, M_obj = params["M0"], params["Me"], params["M_obj"]
a, b = params["a"], params["b"]
T0 = 25.0
y0 = np.array([M0, T0], dtype=float)


def modelo(y, Ta):
    M, T = y
    k = 0.012 * np.exp(0.035 * (T - 50.0))
    dM = -k * (M - Me)
    dT = a * (Ta - T) + b * dM
    return np.array([dM, dT])


def RK4(y, h, Ta):
    k1 = modelo(y, Ta)
    k2 = modelo(y + h * k1 / 2.0, Ta)
    k3 = modelo(y + h * k2 / 2.0, Ta)
    k4 = modelo(y + h * k3, Ta)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simular_lote(Ta, h=2.5, tmax=900):
    n = int(round(tmax / h))
    y = y0.copy()
    tiempos, estados = [0.0], [y.copy()]
    for i in range(n):
        y = RK4(y, h, Ta)
        tiempos.append((i + 1) * h)
        estados.append(y.copy())
    t = np.array(tiempos)
    est = np.array(estados)

    idx = np.where(est[:, 0] <= M_obj)[0]
    if len(idx) == 0:
        return t, est, np.nan, np.nan
    j = idx[0]
    ts = t[j - 1] + (M_obj - est[j - 1, 0]) * (t[j] - t[j - 1]) / (est[j, 0] - est[j - 1, 0])
    T_max = max(np.max(est[:j, 1]), np.interp(ts, t, est[:, 1]))
    return t, est, ts, T_max


def potencia(Ta):
    return P_base * (1.0 + 0.02 * (Ta - 55.0))


# Balance de masa comercial por lote
kg_pulpa = kg_fresca_lote * params["rendimiento_pulpa"]
kg_solido_seco = kg_pulpa / (1.0 + M0)
kg_producto_final = kg_solido_seco * (1.0 + M_obj)
costo_mp_lote = kg_fresca_lote * costo_fruta_kg
ingreso_lote = kg_producto_final * precio_venta_kg

# Evaluación de escenarios 55 - 75 °C
filas = []
for Ta in range(55, 76):
    _, _, ts, T_max = simular_lote(float(Ta))
    P = potencia(Ta)
    E = P * ts / 60.0
    costo_energia_lote = E * tarifa
    costo_total_lote = costo_mp_lote + costo_energia_lote
    utilidad_lote = ingreso_lote - costo_total_lote
    factible = bool(T_max <= T_calidad) if not np.isnan(ts) else False
    filas.append({
        "Temp. Aire (°C)": Ta,
        "Tiempo Lote (min)": ts,
        "Temp. Máx. Fruta (°C)": T_max,
        "Energía/Lote (kWh)": E,
        "Costo Energía/Lote (S/)": costo_energia_lote,
        "Costo Total/kg Seco (S/)": costo_total_lote / kg_producto_final,
        "Utilidad Mensual (S/)": utilidad_lote * lotes_mes,
        "Cumple Calidad": "Sí" if factible else "No (Riesgo térmico)",
        "_factible": factible,
    })

df = pd.DataFrame(filas)
factibles = df[df["_factible"]]

# ---------- Encabezado Principal ----------
st.title("Panel de Optimización Económica y Calidad de Secado")
st.markdown(f"**Receta activa:** `{fruta_sel}` | **Producción neta por lote:** `{kg_producto_final:.2f} kg` de producto terminado")

if not factibles.empty:
    opt = factibles.loc[factibles["Utilidad Mensual (S/)"].idxmax()]
    actual = df[df["Temp. Aire (°C)"] == int(Ta_manual)].iloc[0]
    ahorro_energia_mes = (actual["Costo Energía/Lote (S/)"] - opt["Costo Energía/Lote (S/)"]) * lotes_mes

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperatura Óptima Recomendada", f"{opt['Temp. Aire (°C)']:.0f} °C", f"vs. {Ta_manual:.0f} °C actual")
    c2.metric("Tiempo de Ciclo Óptimo", f"{opt['Tiempo Lote (min)']:.1f} min", f"{opt['Tiempo Lote (min)'] - actual['Tiempo Lote (min)']:.1f} min")
    c3.metric("Costo Unitario (S/ / kg seco)", f"S/ {opt['Costo Total/kg Seco (S/)']:.2f}")
    c4.metric(
        "Utilidad Mensual Proyectada",
        f"S/ {opt['Utilidad Mensual (S/)']:,.0f}",
        f"+S/ {ahorro_energia_mes:,.2f} ahorro energía",
    )
else:
    st.error("⚠️ Ninguna temperatura cumple el límite de calidad configurado. Sube el límite térmico o revisa la receta.")

st.divider()

# ---------- Pestañas Comerciales ----------
tab_negocio, tab_curvas, tab_reporte = st.tabs([
    "📊 Rentabilidad y Punto Óptimo",
    "🌡️ Curva de Proceso en Horno",
    "📋 Orden de Producción y Exportación",
])

with tab_negocio:
    col_g1, col_g2 = st.columns([1.4, 1])
    with col_g1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["Temp. Aire (°C)"],
                y=df["Costo Energía/Lote (S/)"] * lotes_mes,
                mode="lines+markers",
                name="Gasto Eléctrico Mensual (S/)",
                line=dict(color="#1f77b4", width=3),
            )
        )
        fig.add_vline(
            x=T_calidad,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"Límite Calidad ({T_calidad} °C)",
        )
        fig.update_layout(
            title="Impacto de la Temperatura del Horno en el Gasto Eléctrico Mensual",
            xaxis_title="Temperatura programada en horno (°C)",
            yaxis_title="Gasto mensual (S/)",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_g2:
        st.subheader("Resumen Financiero Mensual")
        if not factibles.empty:
            resumen = pd.DataFrame({
                "Concepto": [
                    "Producción mensual (kg)",
                    "Ingresos brutos estimados (S/)",
                    "Costo materia prima (S/)",
                    "Costo energía eléctrica (S/)",
                    "Margen operativo bruto (S/)",
                ],
                "Monto": [
                    f"{kg_producto_final * lotes_mes:,.1f} kg",
                    f"S/ {ingreso_lote * lotes_mes:,.2f}",
                    f"S/ {costo_mp_lote * lotes_mes:,.2f}",
                    f"S/ {opt['Costo Energía/Lote (S/)'] * lotes_mes:,.2f}",
                    f"S/ {opt['Utilidad Mensual (S/)']:,.2f}",
                ],
            })
            st.table(resumen)

with tab_curvas:
    ta_ver = opt["Temp. Aire (°C)"] if not factibles.empty else Ta_manual
    t_sim, est_sim, ts_sim, _ = simular_lote(float(ta_ver))
    mask_plot = t_sim <= (ts_sim * 1.15 if not np.isnan(ts_sim) else 600)

    c_p1, c_p2 = st.columns(2)
    with c_p1:
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=t_sim[mask_plot], y=est_sim[mask_plot, 0], name="Humedad", line=dict(color="#2ca02c", width=3)))
        fig_h.add_hline(y=M_obj, line_dash="dot", annotation_text="Punto de corte comercial")
        fig_h.update_layout(title=f"Curva de Deshidratación a {ta_ver:.0f} °C", xaxis_title="Minutos", yaxis_title="Humedad (b.s.)")
        st.plotly_chart(fig_h, use_container_width=True)

    with c_p2:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=t_sim[mask_plot], y=est_sim[mask_plot, 1], name="Temp. Fruta", line=dict(color="#ff7f0e", width=3)))
        fig_t.add_hline(y=T_calidad, line_dash="dash", line_color="red", annotation_text="Límite de caramelización/calidad")
        fig_t.update_layout(title=f"Temperatura Interna de la Fruta a {ta_ver:.0f} °C", xaxis_title="Minutos", yaxis_title="Temperatura (°C)")
        st.plotly_chart(fig_t, use_container_width=True)

with tab_reporte:
    st.subheader("Tabla de Escenarios Operativos")
    df_mostrar = df.drop(columns=["_factible"])
    st.dataframe(df_mostrar, use_container_width=True)

    csv = df_mostrar.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar Reporte de Escenarios (CSV / Excel)",
        data=csv,
        file_name=f"plan_secado_{fruta_sel.split()[0].lower()}.csv",
        mime="text/csv",
    )
