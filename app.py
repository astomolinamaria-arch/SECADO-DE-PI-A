import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------- Configuración de página ----------
st.set_page_config(
    page_title="Secado de Piña: RK3, RK4 y Optimización",
    page_icon="🍍",
    layout="wide",
)

st.title("🍍 Modelado, Convergencia y Optimización del Secado de Piña")
st.markdown(
    r"""
    Simulación del sistema acoplado de transferencia de masa y calor mediante métodos **Runge-Kutta de orden 3 y 4**:
    $$\frac{dM}{dt} = -k(T)(M - M_e), \qquad \frac{dT}{dt} = a(T_a - T) + b\frac{dM}{dt}, \qquad k(T) = 0.012\,e^{0.035(T - 50)}$$
    """
)

# ---------- Barra lateral: Parámetros interactivos ----------
st.sidebar.header("⚙️ Parámetros del Proceso")

Ta_sim = st.sidebar.slider("Temperatura del aire Ta (°C)", 50.0, 85.0, 65.0, 0.5)
h_sim = st.sidebar.select_slider("Paso de integración h (min)", options=[1.25, 2.5, 5.0, 10.0], value=5.0)
M_obj = st.sidebar.number_input("Humedad objetivo M_obj (kg/kg ss)", 0.05, 1.00, 0.20, 0.01)
tmax = st.sidebar.number_input("Tiempo máximo simulación (min)", 200, 1500, 600, 50)

st.sidebar.header("💰 Parámetros Económicos y Calidad")
P_base = st.sidebar.number_input("Potencia base a 55 °C (kW)", 1.0, 100.0, 12.0, 0.5)
inc_pot = st.sidebar.slider("Incremento potencia por °C (%)", 0.0, 5.0, 2.0, 0.2) / 100.0
tarifa = st.sidebar.number_input("Tarifa eléctrica (S/ por kWh)", 0.10, 5.00, 0.75, 0.05)
lotes = st.sidebar.number_input("Lotes por mes", 1, 2000, 520, 10)
T_CALIDAD = st.sidebar.slider("Límite térmico de calidad (°C)", 55.0, 80.0, 65.0, 0.5)

with st.sidebar.expander("Constantes físicas del modelo"):
    M0 = st.number_input("Humedad inicial M0", 0.5, 10.0, 3.00, 0.1)
    T0 = st.number_input("Temperatura inicial T0 (°C)", 5.0, 50.0, 25.0, 1.0)
    Me = st.number_input("Humedad equilibrio Me", 0.01, 0.50, 0.10, 0.01)
    a = st.number_input("Coeficiente convectivo a", 0.005, 0.200, 0.045, 0.005, format="%.3f")
    b = st.number_input("Acoplamiento evaporativo b", 0.0, 10.0, 3.0, 0.1)

y0 = np.array([M0, T0], dtype=float)


# ---------- Funciones numéricas ----------
def modelo(y, Ta):
    M, T = y
    k = 0.012 * np.exp(0.035 * (T - 50.0))
    dM = -k * (M - Me)
    dT = a * (Ta - T) + b * dM
    return np.array([dM, dT])


def RK3(y, h, Ta):
    k1 = modelo(y, Ta)
    k2 = modelo(y + h * k1 / 2.0, Ta)
    k3 = modelo(y + h * (-k1 + 2.0 * k2), Ta)
    return y + h * (k1 + 4.0 * k2 + k3) / 6.0


def RK4(y, h, Ta):
    k1 = modelo(y, Ta)
    k2 = modelo(y + h * k1 / 2.0, Ta)
    k3 = modelo(y + h * k2 / 2.0, Ta)
    k4 = modelo(y + h * k3, Ta)
    return y + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simular(metodo, h, Ta, t_final=tmax):
    n = int(round(t_final / h))
    y = y0.copy()
    tiempos, estados = [0.0], [y.copy()]
    for i in range(n):
        y = metodo(y, h, Ta)
        tiempos.append((i + 1) * h)
        estados.append(y.copy())
    return np.array(tiempos), np.array(estados)


def tiempo_objetivo(t, y, objetivo=M_obj):
    M = y[:, 0]
    idx = np.where(M <= objetivo)[0]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return t[0]
    t1, t2, M1, M2 = t[j - 1], t[j], M[j - 1], M[j]
    return t1 + (objetivo - M1) * (t2 - t1) / (M2 - M1)


def potencia(Ta):
    return P_base * (1.0 + inc_pot * (Ta - 55.0))


# ---------- Pestañas de la aplicación ----------
tab1, tab2, tab3 = st.tabs([
    "1. Simulación RK3 vs RK4",
    "2. Análisis de Convergencia",
    "3. Optimización Económica y Calidad",
])

# ==================== TAB 1 ====================
with tab1:
    t3, y3 = simular(RK3, h_sim, Ta_sim)
    t4, y4 = simular(RK4, h_sim, Ta_sim)

    col1, col2 = st.columns(2)
    with col1:
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=t3, y=y3[:, 0], mode="lines", name="RK3", line=dict(dash="dash", width=2.5)))
        fig_m.add_trace(go.Scatter(x=t4, y=y4[:, 0], mode="lines", name="RK4", line=dict(width=2.5)))
        fig_m.add_hline(y=M_obj, line_dash="dot", line_color="black", annotation_text=f"M_obj = {M_obj}")
        fig_m.update_layout(
            title=f"Evolución de la Humedad (Ta = {Ta_sim} °C, h = {h_sim} min)",
            xaxis_title="Tiempo (min)",
            yaxis_title="Humedad (kg agua / kg sólido seco)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_m, use_container_width=True)

    with col2:
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=t3, y=y3[:, 1], mode="lines", name="RK3", line=dict(dash="dash", width=2.5)))
        fig_t.add_trace(go.Scatter(x=t4, y=y4[:, 1], mode="lines", name="RK4", line=dict(width=2.5)))
        fig_t.add_hline(y=T_CALIDAD, line_dash="dot", line_color="red", annotation_text=f"Límite calidad = {T_CALIDAD} °C")
        fig_t.update_layout(
            title=f"Evolución de la Temperatura del Producto (Ta = {Ta_sim} °C)",
            xaxis_title="Tiempo (min)",
            yaxis_title="Temperatura (°C)",
            hovermode="x unified",
        )
        st.plotly_chart(fig_t, use_container_width=True)

    st.subheader(f"Comparación Numérica y Económica (Potencia constante = {P_base} kW)")
    filas = []
    for nombre, t, y in [("RK3", t3, y3), ("RK4", t4, y4)]:
        ts = tiempo_objetivo(t, y)
        E = P_base * ts / 60.0 if not np.isnan(ts) else np.nan
        c = E * tarifa if not np.isnan(E) else np.nan
        filas.append([nombre, ts, E, c, c * lotes if not np.isnan(c) else np.nan])

    df_metodos = pd.DataFrame(
        filas,
        columns=["Método", "Tiempo objetivo (min)", "Energía/lote (kWh)", "Costo/lote (S/)", "Costo mensual (S/)"],
    )
    st.dataframe(df_metodos.style.format("{:.4f}", subset=df_metodos.columns[1:]), use_container_width=True)

# ==================== TAB 2 ====================
with tab2:
    st.markdown(f"Comparación frente a solución de referencia **RK4 con h = 0.025 min** a **Ta = {Ta_sim} °C**.")
    t_ref, y_ref = simular(RK4, 0.025, Ta_sim)
    tiempo_ref = tiempo_objetivo(t_ref, y_ref)

    pasos = [10.0, 5.0, 2.5, 1.25]
    datos_error = []
    for metodo, nombre in [(RK3, "RK3"), (RK4, "RK4")]:
        for h in pasos:
            t, y = simular(metodo, h, Ta_sim)
            t_obj = tiempo_objetivo(t, y)
            datos_error.append([
                nombre,
                h,
                t_obj,
                abs(t_obj - tiempo_ref),
                abs(y[-1, 0] - y_ref[-1, 0]),
                abs(y[-1, 1] - y_ref[-1, 1]),
            ])

    df_error = pd.DataFrame(
        datos_error,
        columns=[
            "Método",
            "h (min)",
            "Tiempo objetivo (min)",
            "Error tiempo (min)",
            "Error humedad final",
            "Error temperatura final",
        ],
    )

    col_c1, col_c2 = st.columns([1.3, 1])
    with col_c1:
        fig_conv = go.Figure()
        for nombre in ["RK3", "RK4"]:
            sub = df_error[df_error["Método"] == nombre]
            fig_conv.add_trace(
                go.Scatter(x=sub["h (min)"], y=sub["Error humedad final"], mode="lines+markers", name=nombre)
            )
        fig_conv.update_xaxes(type="log", title="Paso h (min)")
        fig_conv.update_yaxes(type="log", title="Error absoluto de humedad final")
        fig_conv.update_layout(title="Convergencia en escala Log-Log")
        st.plotly_chart(fig_conv, use_container_width=True)

    with col_c2:
        st.markdown("**Orden de convergencia observado** $\log_2(e(h)/e(h/2))$:")
        for nombre in ["RK3", "RK4"]:
            e = df_error[df_error["Método"] == nombre]["Error humedad final"].to_numpy()
            ordenes = np.round(np.log2(e[:-1] / e[1:]), 2)
            st.info(f"**{nombre}** (teórico: {nombre[-1]}): `{ordenes}`")

    st.dataframe(df_error, use_container_width=True)

# ==================== TAB 3 ====================
with tab3:
    temperaturas = np.arange(55, 76, 1)  # Evaluación cada 1 °C entre 55 y 75 °C
    filas_opt = []
    for Ta in temperaturas:
        t, y = simular(RK4, 2.5, float(Ta))
        ts = tiempo_objetivo(t, y)
        P_const = P_base
        E_const = P_const * ts / 60.0
        c_const = E_const * tarifa

        P_var = potencia(Ta)
        E_var = P_var * ts / 60.0
        c_var = E_var * tarifa

        mask = t <= ts
        T_max = max(np.max(y[mask, 1]), np.interp(ts, t, y[:, 1])) if not np.isnan(ts) else np.nan
        factible = bool(T_max <= T_CALIDAD) if not np.isnan(T_max) else False

        filas_opt.append([
            Ta,
            P_var,
            ts,
            T_max,
            E_var,
            c_var,
            c_var * lotes,
            c_const * lotes,
            factible,
        ])

    df_opt = pd.DataFrame(
        filas_opt,
        columns=[
            "Ta (°C)",
            "Potencia var. (kW)",
            "Tiempo (min)",
            "T máx producto (°C)",
            "Energía (kWh)",
            "Costo/lote (S/)",
            "Costo mensual P(Ta) (S/)",
            "Costo mensual P cte (S/)",
            "Factible",
        ],
    )

    factibles = df_opt[df_opt["Factible"]]
    if not factibles.empty:
        mejor = factibles.loc[factibles["Costo mensual P(Ta) (S/)"].idxmin()]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Temperatura Óptima Ta", f"{mejor['Ta (°C)']:.0f} °C")
        m2.metric("Tiempo de Secado", f"{mejor['Tiempo (min)']:.2f} min")
        m3.metric("T máx. Producto", f"{mejor['T máx producto (°C)']:.2f} °C")
        m4.metric("Costo Mensual Óptimo", f"S/ {mejor['Costo mensual P(Ta) (S/)']:,.2f}")
    else:
        st.warning("Ninguna temperatura en el rango 55–75 °C cumple la restricción de calidad actual.")

    fig_opt = go.Figure()
    fig_opt.add_trace(
        go.Scatter(
            x=df_opt["Ta (°C)"],
            y=df_opt["Costo mensual P(Ta) (S/)"],
            mode="lines+markers",
            name="Potencia variable P(Ta)",
        )
    )
    fig_opt.add_trace(
        go.Scatter(
            x=df_opt["Ta (°C)"],
            y=df_opt["Costo mensual P cte (S/)"],
            mode="lines",
            line=dict(dash="dash"),
            name="Potencia constante (12 kW)",
        )
    )
    fig_opt.add_vline(
        x=T_CALIDAD,
        line_dash="dot",
        line_color="red",
        annotation_text=f"Límite calidad ({T_CALIDAD} °C)",
    )
    fig_opt.update_layout(
        title="Costo Mensual vs. Temperatura del Aire",
        xaxis_title="Temperatura del aire Ta (°C)",
        yaxis_title="Costo mensual (S/)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_opt, use_container_width=True)

    st.subheader("Tabla completa de escenarios (55 °C – 75 °C)")
    st.dataframe(
        df_opt.style.format(
            {
                "Potencia var. (kW)": "{:.2f}",
                "Tiempo (min)": "{:.2f}",
                "T máx producto (°C)": "{:.2f}",
                "Energía (kWh)": "{:.2f}",
                "Costo/lote (S/)": "{:.2f}",
                "Costo mensual P(Ta) (S/)": "{:.2f}",
                "Costo mensual P cte (S/)": "{:.2f}",
            }
        ),
        use_container_width=True,
    )
