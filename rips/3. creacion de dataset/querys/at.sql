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
    count(DISTINCT COALESCE(servicio, servicio_desc)) AS servicios_unicos, 
    sum(NULLIF(cantidad,0)) AS cantidad_sum,
    avg(NULLIF(cantidad,0)) AS cantidad_avg,
    stddev(NULLIF(cantidad,0)) AS cantidad_std,
    APPROX_PERCENTILE(NULLIF(cantidad,0), 0.5) AS cantidad_p50,
    -- insumos
    count(DISTINCT iff(tipo_servicio=1,COALESCE(servicio, servicio_desc),NULL)) AS insumos,
    sum(iff(tipo_servicio=1,NULLIF(cantidad,0),NULL)) AS insumo_cantidad_sum,
    avg(iff(tipo_servicio=1,NULLIF(cantidad,0),NULL)) AS insumo_cantidad_avg,
    stddev(iff(tipo_servicio=1,NULLIF(cantidad,0),NULL)) AS insumo_cantidad_std,
    APPROX_PERCENTILE(iff(tipo_servicio=1,NULLIF(cantidad,0),NULL), 0.5) AS insumo_cantidad_p50,
    -- otros
    count(DISTINCT iff(tipo_servicio=2,COALESCE(servicio, servicio_desc),NULL)) AS traslados,
    count(DISTINCT iff(tipo_servicio=3,COALESCE(servicio, servicio_desc),NULL)) AS estasncias,
    count(DISTINCT iff(tipo_servicio=4,COALESCE(servicio, servicio_desc),NULL)) AS honorarios,
FROM VP_INFORMACION.OUTLIERS.RIPS_At__ANON 
group by all