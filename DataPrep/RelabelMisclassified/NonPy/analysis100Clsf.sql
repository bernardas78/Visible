IF OBJECT_ID (N'dbo.f_ind_to_name', N'FN') IS NOT NULL  
    DROP FUNCTION f_ind_to_name;  
GO  
CREATE FUNCTION dbo.f_ind_to_name(@ind int) 
RETURNS varchar(2)
as
begin
 return case @ind
			when 0 then '1'
			when 1 then '2'
			when 2 then '3'
			when 3 then '4'
			when 4 then 'm'
			when 5 then 'ma'
			else 'unknown'
		end
end

go
create table #results (model int,filename varchar(200),actual int,predicted int ,[1] float,[2] float,[3] float,[4] float,m float,ma float,orig_filename varchar(200))

BULK INSERT #results
FROM 'D:\Google Drive\PhD_Data\Visible_ErrorAnalysis\Relabelling\Preds100Clsf_IncRawFilename.csv'
WITH
(
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
)

-- Model accuracies
--select model,count(*), acc=sum(case actual when predicted then 1 else 0 end)/sum(1.) from #results group by model order by model desc

--Most frequetly predicted class and stats
/*
;with predTotals as (
	select 
		filename
		, predicted
		, count(*) as cnt 
	from #results 
	group by filename, predicted)
, maxPreds as (
	select 
		filename
		, predicted
		, cnt as cnt_predicted
		, rank() over (partition by filename order by cnt desc) as rnk
	from predTotals)
, actualEqPreds as (
	select 
		filename
		,actual
		,count(*) as cnt_actual
	from #results
	where actual=predicted
	group by filename, actual
	)
, totalPreds as (
	select 
		filename
		,count(*) as cnt_total
	from #results
	group by filename
	)
select 
	a.filename
	,'=Hyperlink("' + a.filename + '","' + a.filename + '")' as filename_lnk
	, dbo.f_ind_to_name(a.actual) as actual
	--, a.actual
	, dbo.f_ind_to_name(mp.predicted) as most_predicted
	--, mp.predicted
	, a.cnt_actual
	, mp.cnt_predicted
	, tp.cnt_total
	, cast(mp.cnt_predicted as float)/cast(a.cnt_actual as float) as coef
from totalPreds tp
left join maxPreds mp on tp.filename=mp.filename
left join actualEqPreds a on mp.filename=a.filename
where mp.rnk=1
  and cnt_predicted>cnt_actual
--order by predicted desc
order by filename
*/

--Hisghest probability sum
; with pred_score_sums as (
	select 
		--filename 
		orig_filename
		--left ( filename_after_, charindex ('_', filename_after_)-1) as filename
		, dbo.f_ind_to_name(actual) as actual
		, sum(case actual when 0 then [1] when 1 then [2] when 2 then [3] when 3 then [4] when 4 then [m] when 5 then [ma] end) as actual_score
		,sum([1]) as [1]
		,sum([2]) as [2]
		,sum([3]) as [3]
		,sum([4]) as [4]
		,sum(m) as m
		,sum(ma) as ma
		,count(*) as cnt_predicted
	from #results
	group by orig_filename, actual)
, pred_score_sums_unpivot as (
	select
		orig_filename
		, actual
		, actual_score
		, class
		, class_score
		, cnt_predicted
	from 
		(select orig_filename, actual, actual_score, cnt_predicted, [1],[2],[3],[4],[m],[ma] from pred_score_sums) p
	unpivot (class_score for class in ([1],[2],[3],[4],[m],[ma]) )  as unpvt
)	
, pred_high_class as (
	select 
		orig_filename
		, actual
		, actual_score
		, class
		, class_score
		, rank() over (partition by orig_filename order by class_score desc) as rnk
		, cnt_predicted
	from pred_score_sums_unpivot)
, pred_high_class_inc_counts as (select
	orig_filename
	, actual
	, actual_score
	, class as high_class
	, class_score as high_class_score
	, cnt_predicted
	, (class_score-actual_score)/cnt_predicted as avg_over
	, (select count(*) from #results r where r.orig_filename=p.orig_filename and dbo.f_ind_to_name(r.predicted)=p.actual) as cnt_actual
	, (select count(*) from #results r where r.orig_filename=p.orig_filename and dbo.f_ind_to_name(r.predicted)=p.class) as cnt_high_class
from pred_high_class p
where rnk=1
  and actual<>class)
select 
	*
	, cnt_high_class-cnt_actual as cnt_over
from pred_high_class_inc_counts
order by cnt_high_class-cnt_actual --orig_filename
/*select 
	sum([1]) as [1]
	,sum([2]) as [2]
	,sum([3]) as [3]
	,sum([4]) as [4]
	,sum(m) as m
	,sum(ma) as ma
from #results 
where filename='C:\AugmentedForRelabellingMisclassified\1\_100_6547828.jpg' 
--order by predicted
*/
--select * from #results where filename='C:\AugmentedForRelabellingMisclassified\1\_0_9883184.jpg'
/*
select 
dbo.f_ind_to_name(r.actual),
dbo.f_ind_to_name(r.predicted),
* from #results r where orig_filename='D:\Google Drive\PhD_Data\Raw\SCO4\4\992191400000_9_20190907110602485.jpg'
*/
drop table #results
