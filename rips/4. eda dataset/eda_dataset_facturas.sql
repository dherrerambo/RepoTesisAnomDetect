SELECT * FROM VP_INFORMACION.OUTLIERS.DATASET LIMIT 100;

SELECT
	id, count(*) AS reg
FROM VP_INFORMACION.OUTLIERS.DATASET 
GROUP BY 1
ORDER BY 2 DESC
LIMIT 100
;


SELECT 
	APPROX_PERCENTILE(valor_factura,0.001) AS p001,
	APPROX_PERCENTILE(valor_factura,0.01) AS p01,
	APPROX_PERCENTILE(valor_factura,0.05) AS p05,
	APPROX_PERCENTILE(valor_factura,0.1) AS p10,
	APPROX_PERCENTILE(valor_factura,0.2) AS p20,
	APPROX_PERCENTILE(valor_factura,0.3) AS p30,
	APPROX_PERCENTILE(valor_factura,0.5) AS p50,
	APPROX_PERCENTILE(valor_factura,0.8) AS p80,
	APPROX_PERCENTILE(valor_factura,0.9) AS p90,
	APPROX_PERCENTILE(valor_factura,0.99) AS p99,
	APPROX_PERCENTILE(valor_factura,0.995) AS p995
FROM VP_INFORMACION.OUTLIERS.DATASET 
ORDER BY valor_factura ASC


SELECT *, round(reg*1.0/sum(reg) over()*100,2) AS "%" 
FROM (
	SELECT 
		iff(
			valor_factura>=1000
			and personas_unicas>0
			and REG_CONSULTAS+REG_PROCED+REG_URG+REG_HOSP+REG_NAC+REG_MED>0
			, TRUE,FALSE) AS filtro,
		count(*) AS reg
	FROM VP_INFORMACION.OUTLIERS.DATASET 
	GROUP BY 1
)

SELECT * 
FROM VP_INFORMACION.OUTLIERS.DATASET
WHERE valor_factura >= 1000
ORDER BY valor_factura DESC
LIMIT 100

SELECT *
FROM VP_INFORMACION.OUTLIERS.DATASET
WHERE iff(
			valor_factura>=1000
			and personas_unicas>0
			and REG_CONSULTAS+REG_PROCED+REG_URG+REG_HOSP+REG_NAC+REG_MED>0
			, TRUE,FALSE)=true 
LIMIT 100


SELECT *
FROM VP_INFORMACION.OUTLIERS.DF_AF 
WHERE FACTURA='83808C8AD3584A86DB7797E134854104'





SELECT *
FROM VP_INFORMACION.OUTLIERS.DATASET a
WHERE PRESTADOR='0DD90F985DE9C9863A8073BB6BEDBED0'
--WHERE a.PRESTADOR_PROCESADO=TRUE
LIMIT 100
;



UPDATE VP_INFORMACION.OUTLIERS.DATASET a
SET a.PRESTADOR_PROCESADO=TRUE
FROM (
		SELECT DISTINCT PRESTADOR
		FROM VP_INFORMACION.OUTLIERS.DATASET
		WHERE coalesce(LOF, IFOREST, AUTOENCODER) IS NOT NULL
	) b
WHERE a.PRESTADOR=b.PRESTADOR 

UPDATE VP_INFORMACION.OUTLIERS.DATASET a
SET a.PRESTADOR_PROCESADO=FALSE
WHERE a.PRESTADOR_PROCESADO IS NULL




SELECT
	ENSAMBLE 
	, count(*) AS reg
FROM VP_INFORMACION.OUTLIERS.DATASET
WHERE prestador='0DD90F985DE9C9863A8073BB6BEDBED0'
GROUP BY 1

SELECT *
FROM VP_INFORMACION.OUTLIERS.DATASET
WHERE prestador='0DD90F985DE9C9863A8073BB6BEDBED0'
ORDER BY ENSAMBLE__POS asc
LIMIT 100



select *
FROM VP_INFORMACION.OUTLIERS.DATASET
LIMIT 100

SELECT 
	a.PRESTADOR
	, a.PRESTADOR_PROCESADO 
	, count(FACTURA) AS facturas
	, round(sum(VALOR_FACTURA),0) AS valor_factura
	, sum(a.REG_CONSULTAS) as REG_CONSULTAS 
	, sum(a.REG_PROCED) as REG_PROCED 
	, sum(a.REG_URG) as REG_URG 
	, sum(a.REG_HOSP) as REG_HOSP 
	, sum(a.REG_NAC) as REG_NAC 
	, sum(a.REG_MED) as REG_MED
FROM VP_INFORMACION.OUTLIERS.DATASET a
--WHERE PRESTADOR_PROCESADO=true
GROUP BY  ALL
ORDER BY 4 desc