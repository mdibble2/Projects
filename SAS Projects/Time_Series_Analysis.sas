*READING IN DATA;
data stocks;
infile "/folders/myfolders/data/all_stocks_5yr.csv" dsd dlm = ',' truncover;
input date $ open high low close volume name :$4.;
date2 = input(date,MMDDYY8.);
run;

proc contents data= stocks;
run;

*FORMATTING AND CHECKING DATA;
data stocks2;
set stocks;
where name="AAPL";
run;

proc print data = stocks2 (obs=5);
format date2 MMDDYY10.;
run;

*BASIC STATS AND PLOTS;
ods graphics on;
proc timeseries data = stocks2 print = descstats plots = (series corr);
var high;
id date2 interval=day;
label date2=date;
run;


*Trends;
proc timeseries data = stocks2
				out=series
                outtrend=trend;
var low;
id date2 interval=qtr accumulate=avg;
run;

proc print data = trend (obs=4);
title "Trend Statistics";
run;

title1 "Trend Statistics Graph";
proc sgplot data=trend;
   series x=date2 y=max  / lineattrs=(pattern=solid);
   series x=date2 y=mean / lineattrs=(pattern=solid);
   series x=date2 y=min  / lineattrs=(pattern=solid);
   yaxis display=(nolabel);
   format date2 year4.;
   label date2 = date;
run;

*PERIODOGRAM;
proc sort data = stocks;
by date2;
run;

data allstocks;
set stocks;
by date2;

retain hightot;
if first.date2 = 1 then do;
	obs = 0;
	avg = 0;
	hightot = 0;
end;
obs + 1;
hightot = hightot + high;
avg = hightot / obs;
if last.date2 = 1 then output;
run;

proc print data = allstocks (obs=5);
run;

proc timeseries data = allstocks plots = (periodogram);
var avg;
id date2 interval=day;
label date2=date;
run;


proc timeseries data = allstocks plots = (series);
var avg;
where date2<19500;
id date2 interval=day;
label date2=date;
run;


ods graphics off;
