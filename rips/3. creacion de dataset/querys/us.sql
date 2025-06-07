SELECT
    id_carga,
    count(DISTINCT id_persona) AS personas_unicas,
    count(DISTINCT iff(u.sexo='F',u.ID_PERSONA,null)) AS mujeres,
    count(DISTINCT iff(u.sexo='M',u.ID_PERSONA,null)) AS hombres,
    count(DISTINCT iff(u.ZONA_RESIDENCIAL='R',u.id_persona,null)) AS personas_rural,
    count(DISTINCT iff(u.TIPO_USUARIO=1,u.ID_PERSONA,null)) AS personas_c,
    count(DISTINCT iff(u.TIPO_USUARIO=2,u.ID_PERSONA,null)) AS personas_s,
    count(DISTINCT iff(u.TIPO_USUARIO IN (6,7,8),u.ID_PERSONA,null)) AS personas_victima,
    avg(edad) AS edad_avg,
    stddev(edad) AS edad_std,
    approx_percentile(edad,0.5) AS edad_p50,
    count(DISTINCT divipola) AS municipios
FROM VP_INFORMACION.OUTLIERS.v__RIPS_US__ANON u 
GROUP BY all