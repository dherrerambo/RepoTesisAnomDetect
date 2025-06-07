SELECT
    PRESTADOR,
    ID_CARGA,
    FACTURA,
    count(*) AS registros,
    count(DISTINCT id_persona) AS personas_unicas,
    count(DISTINCT iff(sexo='F',id_persona,NULL)) AS mujeres,
    count(DISTINCT iff(edad<18,id_persona,NULL)) AS menores,
    avg(edad) AS edad_avg,
    stddev(edad) AS edad_std,
    approx_percentile(edad,0.5) AS edad_p50,
    count(iff(sexo='F',id_persona,NULL)) AS atenciones_a_mujeres,
    count(iff(edad<18,id_persona,NULL)) AS atenciones_a_menores,
    count(iff(zona='R',ID_PERSONA,NULL)) AS atenciones_a_rurales,
    count(DISTINCT DEPARTAMENTO) AS departamentos,
    count(DISTINCT divipola) AS ciudades,
    count(DISTINCT servicio) AS servicios_unicos,
    sum(iff(AMBITO=1,1,0)) AS ambito_amb,	-- evitar coorelacion perfecta
    sum(iff(AMBITO=2,1,0)) AS ambito_hosp,
    sum(iff(AMBITO=3,1,0)) AS ambito_urg,
    sum(iff(FINALIDAD=1,1,0)) AS finalidad_dx,
    sum(iff(FINALIDAD=2,1,0)) AS finalidad_terapia,
    sum(iff(PERSONA_QUE_ATIENDE IN (1,2),1,0)) AS atencion_por_medico,
    count(DISTINCT DX_PPAL) AS diagnosticos_unicos,
    count(NULLIF(DX_COMPLICACION,'')) AS complicaciones,
    sum(VALOR) AS valor_sum,
    avg(iff(valor=0,NULL,valor)) AS valor_avg
FROM VP_INFORMACION.OUTLIERS.RIPS_AP__ANON 
GROUP BY ALL