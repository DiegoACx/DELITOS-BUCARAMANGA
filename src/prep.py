"""Limpieza replicada del notebook Datos_Delitos_en_Bucaramanga.ipynb (mismo orden de celdas)."""
import pandas as pd

COLS = ["descripcion_conducta", "arma_empleada", "barrios_hecho", "fecha_hecho", "hora_hecho", "edad", "sexo",
        "movil_victima", "movil_agresor", "clase_sitio", "articulo", "delito_solo", "curso_vida", "curso_vida_orden",
        "lapto_tiempo_num", "mes_num", "dia_num", "rango_horario", "tipologia", "rango_horario_orden", "dia_nombre",
        "dia_nombre_orden", "localidad", "num_comuna", "nom_comuna", "cantidad_unica"]

# Diccionario de la celda 91 (la clave duplicada "AGUA CALIENTE" hace que gane la ultima: ACCION FISICA, igual que en el notebook)
ARMA = {"ARMA BLANCA / CORTOPUNZANTE": "ARMA BLANCA", "CORTANTES": "ARMA BLANCA", "CORTOPUNZANTES": "ARMA BLANCA",
        "JERINGA": "ARMA BLANCA", "PUNZANTES": "ARMA BLANCA", "ARMA TRAUMATICA": "ARMA DE FUEGO", "BICICLETA": "VEHICULO",
        "MOTO": "VEHICULO", "SUSTANCIAS TOXICAS": "SUSTANCIAS Y QUIMICOS", "GASES": "SUSTANCIAS Y QUIMICOS",
        "QUIMICOS": "SUSTANCIAS Y QUIMICOS", "ACIDO": "SUSTANCIAS Y QUIMICOS", "ESCOPOLAMINA": "SUSTANCIAS Y QUIMICOS",
        "MEDICAMENTOS": "SUSTANCIAS Y QUIMICOS", "AGUA CALIENTE": "SUSTANCIAS Y QUIMICOS", "VENENO": "SUSTANCIAS Y QUIMICOS",
        "LICOR ADULTERADO": "SUSTANCIAS Y QUIMICOS", "ARTEFACTO INCENDIARIO": "EXPLOSIVO", "PAPA EXPLOSIVA": "EXPLOSIVO",
        "GRANADA DE MANO": "EXPLOSIVO", "PAQUETE BOMBA": "EXPLOSIVO", "COMBUSTIBLE": "EXPLOSIVO",
        "ARTEFACTO EXPLOSIVO/CARGA DINAMITA": "EXPLOSIVO", "LLAMADA TELEFONICA": "COMUNICACION",
        "REDES SOCIALES": "COMUNICACION", "CARTA EXTORSIVA": "COMUNICACION", "AGUA CALIENTE": "ACCION FISICA",
        "SIN EMPLEO DE ARMAS": "ACCION FISICA", "DIRECTA": "ACCION FISICA", "LLAVE MAESTRA": "ACCION FISICA",
        "PALANCAS": "ACCION FISICA", "PRENDAS DE VESTIR": "ACCION FISICA", "BOLSA PLASTICA": "ACCION FISICA",
        "CINTAS/CINTURON": "ACCION FISICA", "CUERDA/SOGA/CADENA": "ACCION FISICA", "CONTUNDENTES": "ACCION FISICA",
        "PERRO": "ACCION FISICA", "NO DISPONIBLE": "NO REPORTADO"}

BARRIOS_FUERA = ['NO DISPONIBLE', 'VILLAS DE SAN IGNACIO (SECTORES BAVARIA I Y II / BETANIA I Y II / INGESER)',
                 'AUTOPISTA F/BLANCA-P/CUESTA.', 'INV. CLUB CHIMITA', 'URBANIZACION CAMPO MADRID', 'LA CEMENTO',
                 'LOS CONQUISTADORES', 'CHIMITA', 'LOS CUADROS', 'BOULEVAR DEL CACIQUE', 'PANTANO II', 'PANTANO III',
                 'URB. ACROPOLIS I', 'BONANZA CAMPESTRE', 'CASCO ANTIGUO', 'VDA. LA ESPERANZA', 'ANILLO VIAL', 'GALLINERAL',
                 'CENTAUROS', 'PEÑON DEL VALLE', 'LA ROSITA', 'SAN PEDRITO', 'BALCÓN DEL LAGO', 'FATIMA', 'MIRADOR DE FATIMA',
                 'LA FLORA', 'LA QUEBRADA', 'LIMONCITO', 'CENFER', 'BOSCONIA', 'ASENTAMIENTO MONEQUE', 'INV. PUNTA BETIN',
                 'VILLA HELENA I', 'VILLA HELENA II', 'INV. LOS CORRALES', 'INV. LUZ DE ESPERANZA', 'NUEVA GRACIA DE DIOS',
                 'PARQUE INDUSTRIAL', 'ROSALTA', 'SAN VALENTÍN', 'VDA. MARCELITAS', 'VDA. RIO DE ORO', 'TEJADOS',
                 'CAMPO ALEGRE II', 'MARIA AUXILIADORA', 'CARRASCO', 'EL UVO', 'GRANJAS REAGAN', 'URB. SAN FERMIN',
                 'VILLA FLOR', 'VILLA REAL DEL SUR', 'VILLA SARA', 'LA GUACAMAYA', 'PRADOS DEL NORTE']

# Regla de rango_edad (pd.cut con right=True, include_lowest=True); la app la reutiliza.
EDAD_BINS = [0, 5, 12, 18, 25, 59, float('inf')]
EDAD_LABELS = ['PRIMERA INFANCIA', 'INFANCIA', 'ADOLESCENCIA', 'JUVENTUD', 'ADULTEZ', 'PERSONA MAYOR']

MOVIL = {"CONDUCTOR MOTOCICLETA": "MOTOCICLETA", "PASAJERO MOTOCICLETA": "MOTOCICLETA", "PASAJERO BUS": "BUS",
         "CONDUCTOR VEHICULO": "VEHICULO", "CONDUCTOR TAXI": "TAXI", "PASAJERO TAXI": "TAXI",
         "PASAJERO VEHICULO": "VEHICULO", "PASAJERO METRO": "METRO", "CONDUCTOR BUS": "BUS"}


def clasificar_hora(h):
    hh = int(str(h)[:2])
    if hh < 6:
        return 'MADRUGADA'
    if hh < 12:
        return 'MAÑANA'
    if hh < 14:
        return 'MEDIODIA'
    if hh < 18:
        return 'TARDE'
    if hh < 20:
        return 'ANOCHECER'
    return 'NOCHE'


def limpiar(csv_path):
    df = pd.read_csv(csv_path, low_memory=False)
    assert df.shape[1] == 26
    df.columns = COLS
    steps = [("filas descargadas", len(df))]
    dfc = df.copy()
    dfc = dfc.drop_duplicates(keep='last')
    steps.append(("tras quitar duplicados", len(dfc)))
    dfc['arma_empleada'] = dfc['arma_empleada'].replace(ARMA)
    dfc = dfc.groupby('arma_empleada').filter(lambda x: len(x) >= 100)
    steps.append(("armas con >=100 casos", len(dfc)))
    dfc = dfc[dfc['arma_empleada'] != 'NO REPORTADO']
    steps.append(("sin arma NO REPORTADO", len(dfc)))
    dfc = dfc[~dfc['nom_comuna'].isin(['CORREGIMIENTO 1', 'CORREGIMIENTO 2', 'CORREGIMIENTO 3'])]
    steps.append(("sin corregimientos", len(dfc)))
    dfc = dfc[~dfc['barrios_hecho'].isin(BARRIOS_FUERA)]
    steps.append(("sin barrios fuera de Bucaramanga", len(dfc)))
    dfc['movil_agresor'] = dfc['movil_agresor'].replace(MOVIL)
    dfc = dfc[~dfc['movil_agresor'].isin(['PASAJERO BARCO', 'PASAJERO AERONAVE', 'TRIPULANTE AERONAVE', 'NO DISPONIBLE'])]
    steps.append(("movil_agresor valido", len(dfc)))
    dfc['movil_victima'] = dfc['movil_victima'].replace(MOVIL)
    dfc = dfc[~dfc['movil_victima'].isin(['NO DISPONIBLE', 'PASAJERO AERONAVE', 'PASAJERO BARCO'])]
    steps.append(("movil_victima valido", len(dfc)))
    dfc = dfc[dfc['sexo'] != 'NO DISPONIBLE']
    steps.append(("sexo distinto de NO DISPONIBLE", len(dfc)))
    dfc['rango_dia'] = dfc['hora_hecho'].apply(clasificar_hora)
    dfc = dfc[dfc['edad'].notna()]
    dfc = dfc[~dfc['edad'].astype(str).isin(['NO DISPONIBLE', '125'])]
    steps.append(("edad valida (sin NA, NO DISPONIBLE, 125)", len(dfc)))
    dfc['edad'] = pd.to_numeric(dfc['edad'], errors='coerce')
    dfc['rango_edad'] = pd.cut(dfc['edad'], bins=EDAD_BINS, labels=EDAD_LABELS,
                               right=True, include_lowest=True)
    n_tip_null = int(dfc['tipologia'].isna().sum())
    dfc = dfc[dfc['tipologia'].notna()]
    steps.append((f"sin tipologia nula ({n_tip_null} nulas)", len(dfc)))
    dfc['anio'] = pd.to_datetime(dfc['fecha_hecho']).dt.year
    dfc['rango_edad'] = dfc['rango_edad'].astype(str)
    return dfc.reset_index(drop=True), steps
