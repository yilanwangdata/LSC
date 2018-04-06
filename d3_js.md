# some d3 code to do visualization 
```
<!DOCTYPE html>
<html lang="en">
	<head>
		<meta charset="utf-8">
		<title>D3.js Essential Training for Data Scientists</title>
		<link rel="stylesheet" type="text/css" href="style.css" >
		<script type="text/javascript" src="d3.v4.js"></script>
    <style>
    g.polygons path {
      fill: white;
      stroke: lightsteelblue;
    }

    g.fuel circle {
      fill: steelblue;
    }

    </style>

	</head>

	<body>
		<script type="text/javascript">
d3.select(window).on("resize",callFunction);
     callFunction();
     function callFunction() {
     var tooltip = d3.select("body").append("div").style("opacity","0").style("position","absolute");
     var svgtest = d3.select("body").select("svg");
			if (!svgtest.empty()) {
				svgtest.remove();
			} 
      var width = window.innerWidth;
      var height = window.innerHeight;


      var vertices = d3.range(100).map(function(d){ return [Math.random()*width,Math.random()*height]; });

      var voronoi = d3.voronoi().size([width,height]);
      var svg = d3.select("body").append("svg").attr("width","100%").attr("height","100%");
      function dragged(){d3.select(this).attr('transform',"translate('+d3.event.x+',"+d3.event.y+")")};
      var chartGroup=svg.append('g').call(d3.drag().on('drag',dragged);
      chartGroup.call(d3.zoom()
           .scaleExtent([0,8,2])
           .on('zoom',function(d){
                 chartGroup.attr('transform',d3.event.transform);
      }));
      chartGroup.append("g").attr("class","polygons")
      .selectAll("path")
        .data(voronoi.polygons(vertices))
        .enter().append("path")
                  .attr("d",function(d){ return "M"+d.join("L")+"Z"; })
                  .on("mousemove",function(d){
											
											tooltip.style("opacity","1")
											.style("left",d[0][0]+"px")
											.style("top",d[0][1]+"px");

											tooltip.html("Number of sides: "+d.length);

      svg.append("g").attr("class","fuel")
      .selectAll("circle")
        .data(vertices)
        .enter().append("circle")
                  .attr("cx",function(d){ return d[0]; })
                  .attr("cy",function(d){ return d[1]; })
                  .attr("r","2.5");
  d3.select("g.polygons").select("path:nth-child(30)")
				.transition().duration(1000)
				.style("fill","blue")
				.attr("transform","translate(10,10)");
  d3.select("g.polygons").select("path:nth-child(30)").dispatch("mousemove");


    </script>
  </body>

</html>

var parseDate = d3.timeParse("%Y");

d3.xml("data2.xml").get(function(error,xml){

  var height = 200;
  var width = 500;
  var margin = {left: 50, right: 50, top: 40, bottom:0};

  xml = [].map.call(xml.querySelectorAll("dat"),function(d){
    return {
      date: parseDate(d.getAttribute("id")),
      top: +d.querySelector("top").textContent,
      middle: +d.querySelector("middle").textContent,
      bottom: +d.querySelector("bottom").textContent

    };

  })
  var x = d3.scaleTime()
            .domain(d3.extent(xml,function(d){return d.date;}))
            .range([0,width]);
  var y = d3.scaleLinear()
            .domain([0,d3.max(xml,function(d){ return d.top+d.middle+d.bottom; })])
            .range([height,0]);

  var categories = ['top','middle','bottom'];

  var stack = d3.stack().keys(categories);

  var area = d3.area()
                .x(function(d,i){ return d.data.date;})
                .y0(function(d){ return y(d[0]);})
                .y1(function(d){ return y(d[1]);});

  var svg = d3.select("body").append("svg").attr("width","100%").attr("height","100%");
  var chartGroup = svg.append("g").attr("transform","translate("+margin.left+","+margin.top+")");
 
  var stacked = stack(xml);
});
  chartGroup.append("g").attr("class","x axis")
                        .attr("transform","translate(0,"+height+")")
                        .call(d3.axisBottom(x));
  chartGroup.append("g").attr("class","y axis")
                        .call(d3.axisLeft(y).ticks(5));
  chartGroup.selectAll("path.area")
    .data(stacked)
    .enter().append("path")
            .attr("class","area")
             .attr("d",function(d){ return area(d); });

chartGroup.selectAll("g.area")
    .data(stacked)
    .enter().append("g")
              .attr("class","area")
    .append("path")
              .attr("class","area")
              .attr("d",function(d){ return area(d); });

var data = [];

data[0] = [];
data[1] = [];
data[2] = [];
data[3] = [];

data[0][0] = [1,2,3];
data[0][1] = [4,5,6];
data[1][0] = [7,8];
data[1][1] = [9,10,11,12];
data[1][2] = [13,14,15];
data[2][0] = [16];
data[3][0] = [17,18];
var width = 1000;
var height = 240;
var barWidth = 100;
var barGap = 10;
var margin = {left:50,right:50,top:0,bottom:0};
var svg = d3.select("body").append("svg").attr("width",width).attr("height",height);
var chartGroup = svg.append("g").attr("transform","translate("+margin.left+","+margin.top+")");
var firstGroups = chartGroup.selectAll("g")
	.data(data)
	.enter().append("g")
		.attr("class",function(d,i){ return "firstLevelGroup"+i; })
		.attr("transform",function(d,i){ return "translate("+(i*(barWidth+barGap))+",0)" ; })

var secondGroups = firstGroups.selectAll("g")
	.data(function(d){ return d;})
	.enter().append("g")
		.attr("class",function(d,i,j){ return "secondLevelGroup"+i; })
		.attr("transform",function(d,i,j){ return "translate(0,"+(height-((i+1)*50))+")"; });

secondGroups.append("rect")
	.attr("x",function(d,i){ return 0;})
	.attr("y","0")
	.attr("width",100)
	.attr("height",50)
	.attr("class","secondLevelRect");


secondGroups.selectAll("circle")
	.data(function(d){ return d; })
	.enter().append("circle")
  	.filter(function(d){ return d>10; })
		.attr("cx",function(d,i){ console.log(d);return ((i*21)+10); })
		.attr("cy","25")
		.attr("r","10")
secondGroups.selectAll("text")
	.data(function(d){ return d; })
	.enter()
.append("text")
	.attr("x",function(d,i){ return ((i*21)+10); })
	.attr("y","25")
	.attr("class","txt")
	.attr("text-anchor","middle")
	.attr("dominant-baseline","middle")
	.text(function(d,i,nodes){console.log(nodes);return d;});
var parseDate = d3.timeParse("%m/%d/%Y");

d3.csv("prices.csv")
    .row(function(d){ return {month: parseDate(d.month), price:Number(d.price.trim().slice(1))}; })
    .get(function(error,data){

var nestedData = d3.nest()
                      .key(function(d){ return d.month.getFullYear(); })
                      .entries(data);

console.log(nestedData);

    });


d3.tsv("data.tsv")
    .row(function(d){ return {month: parseDate(d.month), price:Number(d.price.trim().slice(1))}; })
    .get(function(error,data){
    });

var psv = d3.dsvFormat("|");
d3.text("data.txt")
    .get(function(error,data){

      var rows = psv.parse(data);
      var newRows = [];
      for (var p=0; p<rows.length; p++) {
        newRows.push({month: parseDate(rows[p].month), price:Number(rows[p].price.trim().slice(1))});
      }
    });

d3.xml("data.xml").get(function(error,data){

var xmlLetter = data.documentElement.getElementsByTagName("letter");
var letterNodes = d3.select(data).selectAll("letter")._groups[0][0];
});


d3.text("test.txt").get(function(error, data){

var myTabPositions = [];
var myNewLinePositions = [];

var tabVal = '\\b\t\\b';
var tabMod = 'g';
var tabRegExp = new RegExp(tabVal,tabMod);

var lineVal = '\\b\n\\b';
var lineMod = 'g';
var lineRegExp = new RegExp(lineVal,lineMod);

data.replace(tabRegExp, function(a,b){ myTabPositions.push(b); return a; });
data.replace(lineRegExp, function(a,b){ myNewLinePositions.push(b); return a; });
})

var dataArray = [25,26,28,32,37,45,55,70,90,120,135,150,160,168,172,177,180];
var dataYears = ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016'];

var parseDate = d3.timeParse("%Y");


var height = 200;
var width = 500;

var margin = {left:50,right:50,top:40,bottom:0};

var y = d3.scaleLinear()
            .domain([0,d3.max(dataArray)])
            .range([height,0]);
var x = d3.scaleTime()
            .domain(d3.extent(dataYears,function(d){ return parseDate(d); }))
            .range([0,width]);



var yAxis = d3.axisLeft(y).ticks(3).tickPadding(10).tickSize(10);
var xAxis=d3.axisBottom(x);

var area = d3.area()
                .x(function(d,i){ return x(parseDate(dataYears[i])); })
                .y0(height)
                .y1(function(d){ return y(d); });
var svg = d3.select("body").append("svg").attr("height","100%").attr("width","100%");
var chartGroup = svg.append("g").attr("transform","translate("+margin.left+","+margin.top+")");

chartGroup.append("path").attr("d",area(dataArray));
chartGroup.append("g")
      .attr("class","axis y")
      .call(yAxis);
chartGroup.append('g').attr('class','axis x')
          .attr('transform','translate(0,'+height+')')
          .call(xAxis);

var dataArray = [5,11,18];
var dateDays=['Mon','Wed','Fri'];

var x=d3.scaleBand()
        .domain(dateDays)
        .range([0,170])
        .paddingInner(0.1176);
var xAxis=d3.axisBottom(x);

var svg = d3.select("body").append("svg").attr("height","100%").attr("width","100%");

svg.selectAll("rect")
      .data(dataArray)
      .enter().append("rect")
                .attr("height",function(d,i){ return d*15; })
                .attr("width","50")
                .attr("fill","pink")
                .attr("x",function(d,i){ return 60*i; })
                .attr("y",function(d,i){ return 300-(d*15); });

svg.append('g')
  .attr('class','axis hidden x')
  .attr('transform','translate(0,300)')
  .call(xAxis);
var newX= 300;
svg.selectAll("circle.first")
      .data(dataArray)
      .enter().append("circle")
                .attr("class","first")
                .attr("cx",function(d,i){ newX+=(d*3)+(i*20); return newX; })
                .attr("cy","100")
                .attr("r",function(d){ return d*3; });

var newX = 600;
svg.selectAll("ellipse")
      .data(dataArray)
      .enter().append("ellipse")
                .attr("class","second")
                .attr("cx",function(d,i){ newX+=(d*3)+(i*20); return newX; })
                .attr("cy","100")
                .attr("rx",function(d){ return d*3; })
                .attr("ry","30");

var newX = 900;
svg.selectAll("line")
      .data(dataArray)
      .enter().append("line")
                .attr("x1",newX)
                .attr("stroke-width","2")
                .attr("y1",function(d,i){ return 80+(i*20); })
                .attr("x2",function(d){ return newX+(d*15); })
                .attr("y2",function(d,i){ return 80+(i*20); });

var textArray = ['start','middle','end'];
svg.append("text").selectAll("tspan")
    .data(textArray)
    .enter().append("tspan")
      .attr("x",newX)
      .attr("y",function(d,i){ return 150 + (i*30); })
      .attr("fill","none")
      .attr("stroke","blue")
      .attr("stroke-width","2")
      .attr("dominant-baseline","middle")
      .attr("text-anchor","start")
      .attr("font-size","30")
      .text(function(d){ return d; });

svg.append("line")
      .attr("x1",newX)
      .attr("y1","150")
      .attr("x2",newX)
      .attr("y2","210");

var dataArray = [5,11,18];
var dataDays = ['Mon','Wed','Fri'];

var rainbow=d3.scaleSequential(d3.interpolateRainbow).domain([0,10]);
var rainbow2=d3.scaleSequential(d3.interpolateRainbow).domain([0,3]);
var x = d3.scaleBand()
            .domain(dataDays)
            .range([0,170])
            .paddingInner(0.1176);

var xAxis = d3.axisBottom(x);

var svg = d3.select("body").append("svg").attr("height","100%").attr("width","100%");

var cat20=d3.schemeCategory20;
console.log(cat20);

svg.selectAll("rect")
      .data(dataArray)
      .enter().append("rect")
                .attr("height",function(d,i){ return d*15; })
                .attr("width","50")
                .attr("fill",function(d,i){return rainbow(i);})
                .attr("x",function(d,i){ return 60*i; })
                .attr("y",function(d,i){ return 300-(d*15); });
svg.append("g")
      .attr("class","x axis hidden")
      .attr("transform","translate(0,300)")
      .call(xAxis);

var newX = 300;
svg.selectAll("circle.first")
      .data(dataArray)
      .enter().append("circle")
                .attr("class","first")
                .attr('fill',function(d,i){return rainbow2(i);})
                .attr("cx",function(d,i){ newX+=(d*3)+(i*20); return newX; })
                .attr("cy","100")
                .attr("r",function(d){ return d*3; });

var newX = 600;
svg.selectAll("ellipse")
      .data(dataArray)
      .enter().append("ellipse")
                .attr("class","second")
                .attr('fill',function(d,i){return cat20[i];})
                .attr("cx",function(d,i){ newX+=(d*3)+(i*20); return newX; })
                .attr("cy","100")
                .attr("rx",function(d){ return d*3; })
                .attr("ry","30");

var newX = 900;
svg.selectAll("line")
      .data(dataArray)
      .enter().append("line")
                .attr("x1",newX)
                .attr("stroke-width","2")
                .attr("y1",function(d,i){ return 80+(i*20); })
                .attr("x2",function(d){ return newX+(d*15); })
                .attr("y2",function(d,i){ return 80+(i*20); });

var textArray = ['start','middle','end'];
svg.append("text").selectAll("tspan")
    .data(textArray)
    .enter().append("tspan")
      .attr("x",newX)
      .attr("y",function(d,i){ return 150 + (i*30); })
      .attr("fill","none")
      .attr("stroke","blue")
      .attr("stroke-width","2")
      .attr("dominant-baseline","middle")
      .attr("text-anchor","start")
      .attr("font-size","30")
      .text(function(d){ return d; });

svg.append("line")
      .attr("x1",newX)
      .attr("y1","150")
      .attr("x2",newX)
      .attr("y2","210");

var parseDate=d3.timeParse('%m/%d/%Y');

d3.csv('prices.csv')
     .row(function(d){return {
       month:parseDate(d.month),price:Number(d.price.trim().slice(1))};})
     .get(function(error, data){
  console.log(data);
})

var parseDate = d3.timeParse("%m/%d/%Y");

d3.csv("prices.csv")
    .row(function(d){ return {month: parseDate(d.month), price:Number(d.price.trim().slice(1))}; })
    .get(function(error,data){

      var height=300;
      var width=500;
      var max=d3.max(data,function(d){return d.price;});
      var minDate=d3.min(data,function(d){return d.month;});
      var maxDate=d3.max(data, function(d){return d.month;});
      var y=d3.scaleLinear()
              .domain([0,max])
              .range([height,0]);
      var x=d3.scaleTime()
              .domain([minDate,maxDate])
              .range([0,width]);
      var yAxis=d3.axisLeft(y);
      var xAxis=d3.axisBottom(x);
      var svg=d3.select('body').append('svg').attr('height','100%').attr('width','100%');
      var margin={left:50,right:40,top:40,bottom:0};
      var chartGroup=svg.append('g')
                     .attr('tranform','translate('+margin.left+','+margin.top+')');

      var line=d3.line()
                 .x(function(d){return x(d.month);})
                 .y(function(d){return y(d.price);});
      chartGroup.append('path').attr('d',line(data));

      chartGroup.append('g').attr('class','x axis').call(xAxis)
                .attr('transform','translate(0,'+height+')');
      chartGroup.append('g').attr('class','y axis').call(yAxis);


})

var height=200;
var width=500;
var margin={left:50,right:50,top:40,bottom:0};
var tree=d3.tree().size([width, height]);
var svg=d3.select('body').append('svg').attr('width','100%').attr('height','100%');
var chartGroup=svg.append('g').attr('transform','translate('+margin.left+','+margin.top+')');

d3.json('treeData.json').get(function(error,data){

  var root=d3.hierarchy(data[0]);
  tree(root);
  chartGroup.selectAll('circle')
            .data(root.descendants())
            .enter().append('circle')
            .attr('cx',function(d){return d.x;})
            .attr('cy',function(d){return d.y;})
            .attr('r','5');
chartGroup.selectAll('path')
          .data(root.descendants().slice(1))
          .enter().append('path')
          .attr('class','link')
          .attr('d',function(d){return 'M'+d.x+','+d.y+'L'+d.parent.x+','+d.parant.y ;});

})
```

