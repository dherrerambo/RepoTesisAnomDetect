-- Query de extraccion de informacion del prestador

with dataset AS (
	select
		a.id
		, a.factura
		, a.prestador
		--, o.ips_nom
		, a.id_persona
		, a.tipo_identificacion
		, if(a.regimen='C',1,0) as regimen
		, a.tipo_usuario
	    , a.edad
		, if(a.sexo='F',1,0) as sexo
		, a.departamento
		-- , a.municipio
		, if(a.zona='R',1,0) as zona
	    , fecha
		, year(a.fecha) as anno
		, month(a.fecha) as mes
		, a.servicio, a.servicio_desc 
		, a.servicio_seccion_cups, a.servicio_capitulo_cups
		, a.ambito, a.finalidad
		, a.dx_ppal, a.dx_ppal_desc, a.dx_ppal_capitulo
		, substring(a.dx_ppal,1,3) as cie10_3dig
		, first_value(a.dx_ppal_desc) ignore nulls over(partition by substring(a.dx_ppal,1,3) order by a.dx_ppal) as cie10_3dig_desc
		, coalesce(a.dx_ppal_capitulo = a.dx_rel_capitulo , false) as dx_rel_mismo_capitulo
		, coalesce(a.dx_ppal_capitulo = a.dx_complicacion_capitulo  , false) as dx_complicacion_mismo_capitulo
		, a.forma_realizacion_actoqx
		, a.valor
		, coalesce(a.medicamentos,0) as medicamentos
		-- , coalesce(a.valor_medicamentos,0) as valor_medicamentos
		, coalesce(a.dias_ult_procedimiento,0) as dias_ult_procedimiento
		, coalesce(a.dx_ppal=a.dx_ult_procedimiento,false) as dx_ult_proced
		, coalesce(substring(a.dx_ppal,1,3)=substring(a.dx_ult_procedimiento,1,3),false) as dx_ult_proced_3dig
	from outliers_detection.procedimientos a
		left join outliers_detection.referencias_prestadores o on a.prestador=o._prestador
	where
	    coalesce(a.dx_ppal_desc,'')<>''
	    and coalesce(a.servicio_capitulo_cups,'')<>''
		and coalesce(a.servicio_seccion_cups,'')<>''
	    and a.valor between 1000 and 100000000
	    and a.prestador='517E624BC8E440922E3B77053E805428'
)
SELECT *
FROM dataset