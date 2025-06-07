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
    --
    count(DISTINCT iff(sexo_rn='F',concat_ws('-',id_persona,fecha_nacimiento),null)) AS nacimientos_femenino,
    avg(edad_gestacional) AS edad_gestacional_prom,
    stddev(edad_gestacional) AS edad_gestacional_std,
    avg(peso_rn) AS peso_rn_prom,
    stddev(peso_rn) AS peso_rn_std,
    count(DISTINCT iff(fecha_muerte IS NOT NULL OR causa_basica_muerte IS NOT NULL,id_persona,null)) AS nacido_vivo
FROM VP_INFORMACION.OUTLIERS.RIPS_An__ANON 
GROUP BY ALL