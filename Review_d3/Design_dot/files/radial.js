function d3Radial () {
/* References: 
 https://codepen.io/shellbryson/pen/KzaKLe */

//Read the file
d3.csv("https://people.ischool.berkeley.edu/~niavivek/Design_dot/percent.csv", function(error, data) {
  
    data.forEach(function(d) {
        
        d.skill = +d.skill;
        d.score = parseFloat(+d.score);
        d.topscore= parseFloat(+d.top_score);
        d.covered=parseFloat(+d.covered);
    });
    //Read the file containing the links for suggested resources
  d3.json("https://people.ischool.berkeley.edu/~niavivek/Design_dot/skill_link.json",function(error1, data1) {
      var json_data = data1['Skills'];
 

var end = [];
var end1 = [];
var ind = [];
var start = [];
var count = [];
var count1 = [];
var progress = [];
var top_progress = [];
var step = [];
var step1 = [];


data.forEach(function(d, i) {  

end = d.score;
end1 = d.topscore;
ind = i+1;
start[i] = 0;
count[i] = end;
count1[i] = end1;
progress[i] = start[i];
top_progress[i] = start[i];
step[i] = end[i] < start[i] ? -0.01 : 0.01;
step1[i] = end1[i] < start[i] ? -0.01 : 0.01;
var skill_name = "";
var k;
var skill_ind;

for (k=0,len = json_data.length;k<len;k++){
  if (json_data[k]['ID'] == d.skill){
      skill_name = json_data[k]['Name'];
      skill_ind = k;
  }
}


//Add the radial, link and horizontal bar dynamically depending on the number of skills

document.getElementById("skill").innerHTML += "<div class=\"container\" id=\"container"+ind+"\"></div>";
document.getElementById("container"+ind).innerHTML += "<h2 class=\"skill_name\">"+skill_name+":</h2>";
document.getElementById("container"+ind).innerHTML += "<div class=\"radial\" id=\"radial"+ind+"\"></div>";
document.getElementById("container"+ind).innerHTML += "<div class=\"link\" id=\"link"+ind+"\"></div>";
document.getElementById("container"+ind).innerHTML += "<div class=\"g-bar\" id=\"g-bar"+ind+"\"></div>";
document.getElementById("radial"+ind).innerHTML += "<h3>Mastery Level:</h3><div class=\"help-tip\"><p>This is your skill level based on your performance and behavior in the course.</p></div></br></br>";
document.getElementById("link"+ind).innerHTML += "<h3>Suggested Resources to Improve Mastery Level:<h3></br>";
document.getElementById("link"+ind).innerHTML += "<div class=\"link_container\" id=\"link_container"+ind+"\"></div>";
document.getElementById("radial"+ind).innerHTML += "<div class=\"progress\" id=\"progress"+ind+"\"></div>";
document.getElementById("radial"+ind).innerHTML += "<div class=\"top_progress\" id=\"top_progress"+ind+"\"></div>";
document.getElementById("radial"+ind).innerHTML += "</br><div class=\"progress_score\" id=\"progress_score"+ind+"\"></div>";

document.getElementById("radial"+ind).innerHTML += "<div class=\"top_progress_score\" id=\"top_progress_score"+ind+"\"></div>";
document.getElementById("radial"+ind).innerHTML += "</br></br><div class=\"message\" id=\"message"+ind+"\"></div>";
document.getElementById("progress_score"+ind).innerHTML += "<span class=\"tab\">Your Score</span>";

document.getElementById("top_progress_score"+ind).innerHTML += "<span class=\"tab_avg\">Average Performer Score</span>";
document.getElementById("g-bar"+ind).innerHTML += "<h3>Skill Resources Covered:</h3><div class=\"help-tip\"><p>This bar tells you how much of the course material you have covered.</p></div>";

//Encouraging messages for the students depending on their score
if (d.score >= 70){
document.getElementById("message"+ind).innerHTML += "<span class=\"tab_best\">Way to go!!! You are among the best.</span>";
}
else if (d.score >= 40){
document.getElementById("message"+ind).innerHTML += "<span class=\"tab_better\">Great!! You are right there with others.</span>";
}
else{
document.getElementById("message"+ind).innerHTML += "<span class=\"tab_good\">Keep going. You are doing a good job!</span>";
}

//Displays the links
for (var l=0;l<json_data[skill_ind]['Link_label'].length;l++){
  if (json_data[skill_ind]['Link_type'][l] == "lecture"){
  document.getElementById("link_container"+ind).innerHTML += "<div class=\"icon-display\"><span class=\"icon fa fa-book\" aria-hidden=\"true\"></span>"
  +"<span class=\"tab\"><a href=\""+json_data[skill_ind]['Links'][l]+"\">"+json_data[skill_ind]['Link_label'][l]+"</a></span></div></br></br>";
  }
  else
    {
      document.getElementById("link_container"+ind).innerHTML += "<div class=\"icon-display\"><span class=\"icon fa fa-play\" aria-hidden=\"true\"></span>"
+"<span class=\"tab\"><a href=\""+json_data[skill_ind]['Links'][l]+"\">"+json_data[skill_ind]['Link_label'][l]+"</a></span></div></br></br>";
  }

}


//Add the horizontal bar
var margin = {
    top: 15,
    right: 15,
    bottom: 15,
    left: 20
};
var win_width = window.innerWidth*0.9;
var width = win_width - margin.left - margin.right,
    height = 45 - margin.top - margin.bottom;
//% of window with bar
var f = 0.3;

var svg = d3.select("#g-bar"+ind).append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
    .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
    
    var bars1 = svg.append("rect")
    .attr("class", "bar_1")
    .attr("height", 10)
    .attr("width", f*width)
    .attr('x', 0)
    .attr('y', 0);

  var w = d.covered;
   var bars = svg.append("rect")
    .attr("class", "bar")
    .attr("height", 10)
    .attr("width", w*f*width/100)
    .attr('x', 0)
    .attr('y', 0);
 
//Add legend for the horizontal bar  

var legend1 = svg.append("rect")
    .attr("x", (f+0.03)*width)
    .attr("width", 10)
    .attr("height", 10)
    .style("fill", 'green');

svg.append("text")
    .attr("class","legend")
    .attr("x", (f+0.043)*width)
    .attr("y", 0)
    .text(d.covered +"% Skill resources");

svg.append("text")
    .attr("class","legend")
    .attr("x", (f+0.043)*width)
    .attr("y", 12)
    .text("\t viewed/attempted");

var legend2 = svg.append("rect")
    .attr("x", (f+0.23)*width)
    .attr("width", 10)
    .attr("height", 10)
    .style("fill", 'lightgrey');

svg.append("text")
    .attr("class","legend")
    .attr("x", (f+0.245)*width)
    .attr("y", 0)
    .text((100-d.covered) + "% Skill resources");

svg.append("text")
    .attr("class","legend")
    .attr("x", (f+0.245)*width)
    .attr("y", 12)
    .text("\t not viewed/attempted");

//Add the radial chart
var wrapper = document.getElementById("progress"+ind);
var wrapper1 = document.getElementById("top_progress"+ind);


var radius = 50;
var border = 10;
var strokeSpacing = 0;
var endAngle = Math.PI * 2;
var formatText = d3.format(".0f");
var boxSize = radius * 2;

//Define the circle
var circle = d3.svg.arc()
  .startAngle(0)
  .innerRadius(radius)
  .outerRadius(radius - border);

//setup SVG wrapper
var svg = d3.select(wrapper)
  .append('svg')
  .attr('width', boxSize)
  .attr('height', boxSize);

// ADD Group container
var g = svg.append('g')
  .attr('transform', 'translate(' + boxSize / 2 + ',' + boxSize / 2 + ')');

//Setup track
var track = g.append('g').attr('class', 'radial-progress');
track.append('path')
  .attr('class', 'radial-progress__background')
  .attr('d', circle.endAngle(endAngle));

//Set color of radial chart depending on the score of the student
if (d.score >= 70){
var value = track.append('path').attr('class', 'radial-progress__value_best').attr('d', circle.endAngle(endAngle * count[i]/100));;
}
else if (d.score >= 40){
var value = track.append('path').attr('class', 'radial-progress__value_better').attr('d', circle.endAngle(endAngle * count[i]/100));;
}
else{
var value = track.append('path').attr('class', 'radial-progress__value_good').attr('d', circle.endAngle(endAngle * count[i]/100));;
}
// var value = track.append('path').attr('class', 'radial-progress__value').attr('d', circle.endAngle(endAngle * count[i]/100));;

//Add text value
var numberText = track.append('text')
  .attr('class', 'radial-progress__text')
  .attr('x', '0%')
  .attr('y', '0%')
  .attr('dy', '6px')
  .text(formatText(count[i]));;

//Add the radial chart for the top performer
//Define the circle
var circle1 = d3.svg.arc()
  .startAngle(0)
  .innerRadius(radius)
  .outerRadius(radius - border);

//setup SVG wrapper
var svg1 = d3.select(wrapper1)
  .append('svg')
  .attr('width', boxSize)
  .attr('height', boxSize);

// ADD Group container
var g1 = svg1.append('g')
  .attr('transform', 'translate(' + boxSize / 2 + ',' + boxSize / 2 + ')');

//Setup track
var track1 = g1.append('g').attr('class', 'radial-progress');
track1.append('path')
  .attr('class', 'radial-progress__background')
  .attr('d', circle1.endAngle(endAngle));

//Add colour fill
var value1 = track1.append('path').attr('class', 'radial-progress__value_top').attr('d', circle1.endAngle(endAngle * count1[i]/100));
//Add text value
var numberText1 = track1.append('text')
  .attr('class', 'radial-progress__text')
  .attr('x', '0%')
  .attr('y', '0%')
  .attr('dy', '6px')
  .text(formatText(count1[i]));;


});
});
});
};