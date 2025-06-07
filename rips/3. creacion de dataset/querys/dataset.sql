-- Query creacion de Dataset en base a los RIPS

WITH df AS (
	SELECT
		-- factura (af)
			af.FACTURA, af.ID_CARGA,		-- llave
			af.PRESTADOR,
			af.FECHA_EXP_FACTURA,
			af.FECHA_INICIO,
			af.FECHA_FIN,
			af.REGIMEN,
			af.CONTRATO,
			af.VALOR_TOTAL_COPAGO, af.VALOR_COMISION, af.VALOR_DESCUENTOS, af.VALOR_NETO,
		-- personas (us)
			--us.ID_CARGA,
			us.PERSONAS_UNICAS,
			us.MUJERES, us.HOMBRES,
			us.PERSONAS_RURAL, us.PERSONAS_C, --us.PERSONAS_S,
			us.PERSONAS_VICTIMA,
			us.EDAD_AVG, us.EDAD_STD, us.EDAD_P50,
			--us.MUNICIPIOS
		-- consulta (ac)
			--ac.PRESTADOR, ac.ID_CARGA, ac.FACTURA,
			ac.REGISTROS AS reg_consultas,
			ac.PERSONAS_UNICAS AS personas_unicas_con_consultas,
			ac.MUJERES AS mujeres_con_consultas,
			ac.MENORES AS menores_con_consultas,
			ac.EDAD_AVG AS avg_personas_con_consultas,
			--ac.EDAD_STD, ac.EDAD_P50,
			ac.ATENCIONES_A_MUJERES AS num_consultas_mujeres,
			ac.ATENCIONES_A_MENORES AS num_consultas_menores,
			ac.ATENCIONES_A_RURALES AS num_consultas_rurales,
			ac.ATENCIONES_CON_DX_CONFIRMADO AS num_consultas_dx_conf,
			--ac.DEPARTAMENTOS, ac.CIUDADES,
			ac.SERVICIOS_UNICOS AS cod_consultas_unicos,
			ac.DIAGNOSTICOS_UNICOS AS dx_consulta_unicos,
			ac.VALOR_SUM AS sum_valor_consultas,
			ac.VALOR_AVG AS avg_por_consulta,
		-- procedimientos (ap)
			--ap.PRESTADOR, ID_CARGA, FACTURA,
			ap.REGISTROS AS reg_proced,
			ap.PERSONAS_UNICAS AS personas_unicas_con_proced, 
			ap.MUJERES AS mujeres_con_proced,
			ap.MENORES AS menores_con_proced,
			ap.EDAD_AVG AS avg_personas_con_proced,
			--ap.EDAD_STD, ap.EDAD_P50,
			ap.ATENCIONES_A_MUJERES AS num_proced_mujeres,
			ap.ATENCIONES_A_MENORES AS num_proced_menores,
			ap.ATENCIONES_A_RURALES AS num_proced_rurales,
			--ap.DEPARTAMENTOS, ap.CIUDADES,
			ap.SERVICIOS_UNICOS AS cod_proced_unicos,
			--ap.AMBITO_AMB,
			ap.AMBITO_HOSP AS proced_ambito_hosp,
			ap.AMBITO_URG AS proced_ambito_urg,
			ap.FINALIDAD_DX AS proced_para_dx,
			ap.FINALIDAD_TERAPIA AS proced_para_terapia,
			ap.ATENCION_POR_MEDICO AS proced_medico,
			--ap.DIAGNOSTICOS_UNICOS,
			ap.COMPLICACIONES AS proced_con_complicaciones,
			ap.VALOR_SUM AS sum_valor_proced,
			ap.VALOR_AVG AS avg_valor_prom,
		-- urgencias (au)
			--au.PRESTADOR, ID_CARGA, FACTURA,
			au.REGISTROS AS reg_urg,
			au.PERSONAS_UNICAS AS personas_unicas_con_urg, 
			au.MUJERES AS mujeres_con_urg,
			au.MENORES AS menores_con_urg,
			au.EDAD_AVG AS avg_personas_con_urg,
			--EDAD_STD, EDAD_P50,
			au.ATENCIONES_A_MUJERES AS num_urg_mujeres,
			au.ATENCIONES_A_MENORES AS num_urg_menores,
			au.ATENCIONES_A_RURALES AS num_urg_rurales,
			--DEPARTAMENTOS, CIUDADES,
			au.INGRESOS_DE_URGENCIA AS num_ingresos_urg,
			ESTANCIA_URGENCIA_PROM AS avg_estancia_urg,
			--ESTANCIA_URGENCIA_STD, ESTANCIA_URGENCIA_P50,
			au.PERSONAS_FALLECIDAS AS personas_fallecidas_urg,
		-- estancias hospitalarias (ah)
			--ah.PRESTADOR, ah.ID_CARGA, ah.FACTURA
			ah.REGISTROS AS reg_hosp,
			ah.PERSONAS_UNICAS AS personas_unicas_con_hosp, 
			ah.MUJERES AS mujeres_con_hosp,
			ah.MENORES AS menores_con_hosp,
			ah.EDAD_AVG AS avg_personas_con_hosp,
			--EDAD_STD, EDAD_P50,
			ah.ATENCIONES_A_MUJERES AS num_hosp_mujeres,
			ah.ATENCIONES_A_MENORES AS num_hosp_menores,
			ah.ATENCIONES_A_RURALES AS num_hosp_rurales,
			--ah.DEPARTAMENTOS ,ah.CIUDADES ,
			ah.INGRESOS_DE_HOSPITALARIOS AS num_hosp,
			ah.ESTANCIA_HOSP_PROM AS avg_estancia_hosp,
			ah.ESTANCIA_HOSP_STD AS std_estancia_hosp,
			ah.ESTANCIA_HOSP_P50 AS p50_estancia_hosp,
			ah.PERSONAS_FALLECIDAS AS personas_fallecidas_hosp,
		-- nacimientos (an)
			--PRESTADOR, an.ID_CARGA, an.FACTURA
			an.REGISTROS AS reg_nac,
			an.PERSONAS_UNICAS AS personas_unicas_con_nac, 
			an.MUJERES AS mujeres_con_nac,
			an.MENORES AS menores_con_nac,
			an.EDAD_AVG AS avg_personas_con_nac,
			--EDAD_STD, EDAD_P50,
			an.ATENCIONES_A_MUJERES AS num_nac_mujeres,
			an.ATENCIONES_A_MENORES AS num_nac_menores,
			an.ATENCIONES_A_RURALES AS num_nac_rurales,
			--an.DEPARTAMENTOS, an.CIUDADES
			an.NACIMIENTOS_FEMENINO AS num_rn_mujeres,
			an.EDAD_GESTACIONAL_PROM AS avg_edad_gest,
			an.EDAD_GESTACIONAL_STD AS std_edad_gest,
			an.PESO_RN_PROM AS avg_peso_rn,
			an.PESO_RN_STD AS std_peso_rn,
			an.NACIDO_VIVO AS num_nac_vivo,
		-- medicamentos (am)
			--am.PRESTADOR, am.ID_CARGA, am.FACTURA
			am.REGISTROS AS reg_med,
			am.PERSONAS_UNICAS AS personas_unicas_con_med, 
			am.MUJERES AS mujeres_con_med,
			am.MENORES AS menores_con_med,
			am.EDAD_AVG AS avg_personas_con_med,
			--EDAD_STD, EDAD_P50,
			am.ATENCIONES_A_MUJERES AS num_med_mujeres,
			am.ATENCIONES_A_MENORES AS num_med_menores,
			am.ATENCIONES_A_RURALES AS num_med_rurales,
			--am.DEPARTAMENTOS, am.CIUDADES
			am.MEDICAMENTOS AS num_med,
			am.MEDICAMENTOS_NOPBS AS num_med_nopbs,
			am.PBS_UND_SUM AS med_pbs_cantidad,
			am.PBS_UND_AVG AS avg_med_pbs_cantidad,
			am.PBS_UND_STD AS std_med_pbs_cantidad,
			am.PBS_UND_P50 AS p50_med_pbs_cantidad,
			am.PBS_VALOR_SUM AS med_pbs_valor,
			am.PBS_VALOR_AVG AS avg_med_pbs_valor,
			am.PBS_VALOR_STD AS std_med_pbs_valor,
			am.PBS_VALOR_P50 AS p50_med_pbs_valor,
			am.NOPBS_UND_SUM AS med_nopbs_cantidad,
			am.NOPBS_UND_AVG AS avg_med_nopbs_cantidad,
			am.NOPBS_UND_STD AS std_med_nopbs_cantidad,
			am.NOPBS_UND_P50 AS p50_med_nopbs_cantidad,
			am.NOPBS_VALOR_SUM AS med_nopbs_valor,
			am.NOPBS_VALOR_AVG AS avg_med_nopbs_valor,
			am.NOPBS_VALOR_STD AS std_med_nopbs_valor,
			am.NOPBS_VALOR_P50 AS p50_med_nopbs_valor,
	FROM VP_INFORMACION.outliers.df_af af
		LEFT JOIN VP_INFORMACION.outliers.df_us us ON af.id_carga=us.id_carga
		LEFT JOIN VP_INFORMACION.outliers.df_ac ac ON af.id_carga=ac.id_carga AND af.factura=ac.factura
		LEFT JOIN VP_INFORMACION.outliers.df_ap ap ON af.id_carga=ap.id_carga AND af.factura=ap.factura
		LEFT JOIN VP_INFORMACION.outliers.df_au au ON af.id_carga=au.id_carga AND af.factura=au.factura
		LEFT JOIN VP_INFORMACION.outliers.df_ah ah ON af.id_carga=ah.id_carga AND af.factura=ah.factura
		LEFT JOIN VP_INFORMACION.outliers.df_an an ON af.id_carga=an.id_carga AND af.factura=an.factura
		LEFT JOIN VP_INFORMACION.outliers.df_am am ON af.id_carga=am.id_carga AND af.factura=am.factura
)
SELECT
	concat_ws('-',a.FACTURA, a.ID_CARGA) AS id
	, a.FACTURA
	, a.ID_CARGA
	, a.PRESTADOR
	-- , COALESCE(a.FECHA_EXP_FACTURA, a.FECHA_FIN) AS FECHA_EXP_FACTURA 
	, year(a.fecha_fin) AS periodo_fin_ano
	, month(a.fecha_fin) AS periodo_fin_mes
	, FECHA_INICIO
	, fecha_fin
	, (iff(a.FECHA_INICIO=a.FECHA_FIN,1,datediff('day',a.FECHA_INICIO,a.FECHA_FIN))/365.25)*12 AS duracion_facturacion
	, iff(a.REGIMEN='C',1,0) AS regimen
	, COALESCE(a.CONTRATO,'NA') AS contrato
	, coalesce(a.VALOR_TOTAL_COPAGO,0) as valor_copago
	, coalesce(a.VALOR_COMISION,0) as valor_comision
	, coalesce(a.VALOR_DESCUENTOS,0) as valor_descuentos
	, coalesce(a.VALOR_NETO,0) as valor_factura
	, coalesce(a.PERSONAS_UNICAS,0) as personas_unicas
	, coalesce(a.MUJERES,0) as mujeres
	, coalesce(a.HOMBRES,0) as hombres
	, coalesce(a.PERSONAS_RURAL,0) as personas_rural
	, coalesce(a.PERSONAS_C,0) as personas_c
	, coalesce(a.PERSONAS_VICTIMA,0) as personas_victima
	, coalesce(a.EDAD_AVG,0) as edad_avg
	, coalesce(a.EDAD_STD,0) as edad_std
	, coalesce(a.EDAD_P50,0) as edad_p50
	, coalesce(a.REG_CONSULTAS,0) as reg_consultas
	, coalesce(a.PERSONAS_UNICAS_CON_CONSULTAS,0) as personas_unicas_con_consultas
	, coalesce(a.MUJERES_CON_CONSULTAS,0) as mujeres_con_consultas
	, coalesce(a.MENORES_CON_CONSULTAS,0) as menores_con_consultas
	, coalesce(a.AVG_PERSONAS_CON_CONSULTAS,0) as avg_personas_con_consultas
	, coalesce(a.NUM_CONSULTAS_MUJERES,0) as num_consultas_mujeres
	, coalesce(a.NUM_CONSULTAS_MENORES,0) as num_consultas_menores
	, coalesce(a.NUM_CONSULTAS_RURALES,0) as num_consultas_rurales
	, coalesce(a.NUM_CONSULTAS_DX_CONF,0) as num_consultas_dx_conf
	, coalesce(a.COD_CONSULTAS_UNICOS,0) as cod_consultas_unicos
	, coalesce(a.DX_CONSULTA_UNICOS,0) as dx_consulta_unicos
	, coalesce(a.SUM_VALOR_CONSULTAS,0) as sum_valor_consultas
	, coalesce(a.AVG_POR_CONSULTA,0) as avg_por_consulta
	, coalesce(a.REG_PROCED,0) as reg_proced
	, coalesce(a.PERSONAS_UNICAS_CON_PROCED,0) as personas_unicas_con_proced
	, coalesce(a.MUJERES_CON_PROCED,0) as mujeres_con_proced
	, coalesce(a.MENORES_CON_PROCED,0) as menores_con_proced
	, coalesce(a.AVG_PERSONAS_CON_PROCED,0) as avg_personas_con_proced
	, coalesce(a.NUM_PROCED_MUJERES,0) as num_proced_mujeres
	, coalesce(a.NUM_PROCED_MENORES,0) as num_proced_menores
	, coalesce(a.NUM_PROCED_RURALES,0) as num_proced_rurales
	, coalesce(a.COD_PROCED_UNICOS,0) as cod_proced_unicos
	, coalesce(a.PROCED_AMBITO_HOSP,0) as proced_ambito_hosp
	, coalesce(a.PROCED_AMBITO_URG,0) as proced_ambito_urg
	, coalesce(a.PROCED_PARA_DX,0) as proced_para_dx
	, coalesce(a.PROCED_PARA_TERAPIA,0) as proced_para_terapia
	, coalesce(a.PROCED_MEDICO,0) as proced_medico
	, coalesce(a.PROCED_CON_COMPLICACIONES,0) as proced_con_complicaciones
	, coalesce(a.SUM_VALOR_PROCED,0) as sum_valor_proced
	, coalesce(a.AVG_VALOR_PROM,0) as avg_valor_prom
	, coalesce(a.REG_URG,0) as reg_urg
	, coalesce(a.PERSONAS_UNICAS_CON_URG,0) as personas_unicas_con_urg
	, coalesce(a.MUJERES_CON_URG,0) as mujeres_con_urg
	, coalesce(a.MENORES_CON_URG,0) as menores_con_urg
	, coalesce(a.AVG_PERSONAS_CON_URG,0) as avg_personas_con_urg
	, coalesce(a.NUM_URG_MUJERES,0) as num_urg_mujeres
	, coalesce(a.NUM_URG_MENORES,0) as num_urg_menores
	, coalesce(a.NUM_URG_RURALES,0) as num_urg_rurales
	, coalesce(a.NUM_INGRESOS_URG,0) as num_ingresos_urg
	, coalesce(a.AVG_ESTANCIA_URG,0) as avg_estancia_urg
	, coalesce(a.PERSONAS_FALLECIDAS_URG,0) as personas_fallecidas_urg
	, coalesce(a.REG_HOSP,0) as reg_hosp
	, coalesce(a.PERSONAS_UNICAS_CON_HOSP,0) as personas_unicas_con_hosp
	, coalesce(a.MUJERES_CON_HOSP,0) as mujeres_con_hosp
	, coalesce(a.MENORES_CON_HOSP,0) as menores_con_hosp
	, coalesce(a.AVG_PERSONAS_CON_HOSP,0) as avg_personas_con_hosp
	, coalesce(a.NUM_HOSP_MUJERES,0) as num_hosp_mujeres
	, coalesce(a.NUM_HOSP_MENORES,0) as num_hosp_menores
	, coalesce(a.NUM_HOSP_RURALES,0) as num_hosp_rurales
	, coalesce(a.NUM_HOSP,0) as num_hosp
	, coalesce(a.AVG_ESTANCIA_HOSP,0) as avg_estancia_hosp
	, coalesce(a.STD_ESTANCIA_HOSP,0) as std_estancia_hosp
	, coalesce(a.P50_ESTANCIA_HOSP,0) as p50_estancia_hosp
	, coalesce(a.PERSONAS_FALLECIDAS_HOSP,0) as personas_fallecidas_hosp
	, coalesce(a.REG_NAC,0) as reg_nac
	, coalesce(a.PERSONAS_UNICAS_CON_NAC,0) as personas_unicas_con_nac
	, coalesce(a.MUJERES_CON_NAC,0) as mujeres_con_nac
	, coalesce(a.MENORES_CON_NAC,0) as menores_con_nac
	, coalesce(a.AVG_PERSONAS_CON_NAC,0) as avg_personas_con_nac
	, coalesce(a.NUM_NAC_MUJERES,0) as num_nac_mujeres
	, coalesce(a.NUM_NAC_MENORES,0) as num_nac_menores
	, coalesce(a.NUM_NAC_RURALES,0) as num_nac_rurales
	, coalesce(a.NUM_RN_MUJERES,0) as num_rn_mujeres
	, coalesce(a.AVG_EDAD_GEST,0) as avg_edad_gest
	, coalesce(a.STD_EDAD_GEST,0) as std_edad_gest
	, coalesce(a.AVG_PESO_RN,0) as avg_peso_rn
	, coalesce(a.STD_PESO_RN,0) as std_peso_rn
	, coalesce(a.NUM_NAC_VIVO,0) as num_nac_vivo
	, coalesce(a.REG_MED,0) as reg_med
	, coalesce(a.PERSONAS_UNICAS_CON_MED,0) as personas_unicas_con_med
	, coalesce(a.MUJERES_CON_MED,0) as mujeres_con_med
	, coalesce(a.MENORES_CON_MED,0) as menores_con_med
	, coalesce(a.AVG_PERSONAS_CON_MED,0) as avg_personas_con_med
	, coalesce(a.NUM_MED_MUJERES,0) as num_med_mujeres
	, coalesce(a.NUM_MED_MENORES,0) as num_med_menores
	, coalesce(a.NUM_MED_RURALES,0) as num_med_rurales
	, coalesce(a.NUM_MED,0) as num_med
	, coalesce(a.NUM_MED_NOPBS,0) as num_med_nopbs
	, coalesce(a.MED_PBS_CANTIDAD,0) as med_pbs_cantidad
	, coalesce(a.AVG_MED_PBS_CANTIDAD,0) as avg_med_pbs_cantidad
	, coalesce(a.STD_MED_PBS_CANTIDAD,0) as std_med_pbs_cantidad
	, coalesce(a.P50_MED_PBS_CANTIDAD,0) as p50_med_pbs_cantidad
	, coalesce(a.MED_PBS_VALOR,0) as med_pbs_valor
	, coalesce(a.AVG_MED_PBS_VALOR,0) as avg_med_pbs_valor
	, coalesce(a.STD_MED_PBS_VALOR,0) as std_med_pbs_valor
	, coalesce(a.P50_MED_PBS_VALOR,0) as p50_med_pbs_valor
	, coalesce(a.MED_NOPBS_CANTIDAD,0) as med_nopbs_cantidad
	, coalesce(a.AVG_MED_NOPBS_CANTIDAD,0) as avg_med_nopbs_cantidad
	, coalesce(a.STD_MED_NOPBS_CANTIDAD,0) as std_med_nopbs_cantidad
	, coalesce(a.P50_MED_NOPBS_CANTIDAD,0) as p50_med_nopbs_cantidad
	, coalesce(a.MED_NOPBS_VALOR,0) as med_nopbs_valor
	, coalesce(a.AVG_MED_NOPBS_VALOR,0) as avg_med_nopbs_valor
	, coalesce(a.STD_MED_NOPBS_VALOR,0) as std_med_nopbs_valor
	, coalesce(a.P50_MED_NOPBS_VALOR,0) as p50_med_nopbs_valor
FROM df a