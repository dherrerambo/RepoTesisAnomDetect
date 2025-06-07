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
    count(DISTINCT concat_ws('-',id_persona,fecha_ingreso,prestador)) AS ingresos_de_urgencia,
    avg(datediff('day', fecha_ingreso, fecha_salida)) AS estancia_urgencia_prom,
    stddev(datediff('day', fecha_ingreso, fecha_salida)) AS estancia_urgencia_std,
    APPROX_PERCENTILE(datediff('day', fecha_ingreso, fecha_salida),0.5) AS estancia_urgencia_p50,
    count(DISTINCT iff(estado_salida=1,id_persona,null)) AS personas_fallecidas
FROM VP_INFORMACION.OUTLIERS.RIPS_AU__ANON 
GROUP BY ALL