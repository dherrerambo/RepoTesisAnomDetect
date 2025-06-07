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
    count(DISTINCT dx) AS diagnosticos_unicos,
    sum(iff(tipo_dx_ppal>1,1,0)) AS atenciones_con_dx_confirmado,
    sum(VALOR_CONSULTA) AS valor_sum,
    avg(iff(VALOR_CONSULTA=0,NULL,VALOR_CONSULTA)) AS valor_avg
FROM VP_INFORMACION.OUTLIERS.RIPS_AC__ANON 
GROUP BY ALL