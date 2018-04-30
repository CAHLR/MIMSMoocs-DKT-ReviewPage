
function d3progress (collapse){

    // Remove the previous canvas
d3.selectAll('.course_progress svg').remove();
d3.selectAll('.legend1 svg').remove();

//Get which radio button is selected
var selection = document.getElementsByName('module');
var selected_value;
for(var i = 0; i < selection.length; i++){
    if(selection[i].checked){
        selected_value = selection[i].value;
        document.getElementById("module_name").innerHTML =""
        break;
    }
}
/* Reference: https://www.w3schools.com/howto/howto_js_collapsible.asp*/

//Collapse/uncollapse the course progress section on click
if (collapse == true){
var coll = document.getElementsByClassName("collapsible_1");
var i;

for (i = 0; i < coll.length; i++) {
  coll[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.maxHeight){
      content.style.maxHeight = null;

    } else {
      content.style.maxHeight = content.scrollHeight + "px";
    } 
  });
}
}


//Set margin, width and height of svg
var win_width = window.innerWidth*0.7;
var margin = {top: 30, right: 20, bottom: 10, left: 10},
    width = win_width - margin.left - margin.right,
    height = 150 - margin.top - margin.bottom;


// Set the ranges
var x = d3.time.scale().range([0.25*width, 0.75*width]);
var y = d3.scale.linear().range([height, 0]);


// Define the line
var valueline = d3.svg.line()
    .x(function(d) { return x(d.number); })
    .y(function(d) { return y(d.length); })
                .interpolate("linear");


// Adds the svg canvas
var svg = d3.select(".course_progress")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);
var course_progress = svg.append("g")
        .attr("transform", 
              "translate(" + margin.left + "," + margin.top + ")");

//Add legend for course progress bar
var svg1 = d3.select(".legend1")
    .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", 30 + 5 + 5);

var legend = svg1.append("g")
        .attr("transform", 
              "translate(" + (margin.left+10) + "," + 5 + ")");
   

var legend1 = legend.append('svg:foreignObject')
    .attr("x", 0.35*width)
    .attr("width", 40)
    .attr("height", 40)
    .html('<small><i class="fa fa-check-circle fa-2x"></i></small>')
    .attr("color",'green');

legend.append("text")
    .attr("class","legend")
    .attr("x", 0.35*width+30)
    .attr("y", 18)
    .text("Completed");


var legend2 = legend.append('svg:foreignObject')
.attr("x", 0.48*width)
.attr("width", 40)
    .attr("height", 40)
    .html('<small><i class="fa fa-circle fa-2x"></i></small>')
    .attr("color",'lightgrey');

legend.append("text")
    .attr("class","legend")
    .attr("x", 0.48*width+30)
    .attr("y", 18)
    .text("Remaining");

var legend3 = legend.append('svg:foreignObject')
.attr("x", 0.6*width)
.attr("width", 40)
    .attr("height", 40)
    .html('<small><i class="fa fa-circle-thin fa-2x"></i></small>')
    .attr("color",'black');

legend.append("text")
    .attr("class","legend")
    .attr("x", 0.6*width+30)
    .attr("y", 18)
    .text("Skipped");

//Read the file and depending on which radio button is selected populate the values.
//Index with [] refers to the module/section number
var course_name,module_name,section_name;

d3.json("https://people.ischool.berkeley.edu/~niavivek/Design_dot/modules.json",function(error, json_data) {
// $.getJSON("https://people.ischool.berkeley.edu/~niavivek/Design_dot/modules.json", function(json_data) {
    var json_data = json_data['Modules'];
    if (selected_value == "module"){
      
      data = json_data;
//Module view
       data.forEach(function(d, i) {
        if (d['here'] == "True"){
            course_name = "Asynchronous Programming with Javascript";
            module_name = json_data[0]['title'];
            section_name = json_data[0]['Section'][7]['title'];
        }
        d.number = i+1;
        d.fill = d['visited'];

        if (i == 0){
            d.length = 2;
        }
        else{
        d.length = +2;
    }
});
   }

// // Section View
else if (selected_value == "section"){
        data = json_data[0]['Section'];
       data.forEach(function(d, i) {
        if (d['here'] == "True"){
            course_name = "Asynchronous Programming with Javascript";
            module_name = json_data[0]['title'];
            section_name = json_data[0]['Section'][7]['title'];
        }
        
        d.number = i+1;
        d.fill = d['visited'];
        if (i == 0){
            d.length = 2;
        }
        else{
        d.length = +2;
    }
});
   }
   else{

// //Sub-Section View
        var data = json_data[0]['Section'][7]['Sub-Sections'];
       data.forEach(function(d, i) {
        if (d['here'] == "True"){
            course_name = "Asynchronous Programming with Javascript";
            module_name = json_data[0]['title'];
            section_name = json_data[0]['Section'][7]['title'];
        }
        d.number = i+1;
        
        d.fill = d['visited'];
        if (i == 0){
            d.length = 2;
        }
        else{
        d.length = +2;
    }
});
}

//Shorten the bar lengths if the number of modules/sections are 1/2/3
if (data.length == 1){
    // Set the ranges
x = d3.time.scale().range([0.5*width, 0.5*width]);
y = d3.scale.linear().range([height, 0]);
}

if (data.length == 2){
    // Set the ranges
x = d3.time.scale().range([0.42*width, 0.55*width]);
y = d3.scale.linear().range([height, 0]);
}

if (data.length == 3){
    // Set the ranges
x = d3.time.scale().range([0.38*width, 0.6*width]);
y = d3.scale.linear().range([height, 0]);
}

change_name();
//Function to change course/module/section name based on radio button selection
function change_name(){
if (selected_value == "module"){
document.getElementById("module_name").innerHTML += "<h3 class=\"collap_header\">Course: "+course_name+"</h3>";
}
else if (selected_value == "section"){
document.getElementById("module_name").innerHTML += "<h3 class=\"collap_header\">Module Name: "+module_name+"</h3>";
}
else{
document.getElementById("module_name").innerHTML += "<h3 class=\"collap_header\">Section Name: "+section_name+"</h3>";
}
};

//Add tooltip for the course progress bar
var tooltip = d3.select("body")
  .append("div")
  .attr('class', 'tooltip_1');

    x.domain(d3.extent(data, function(d) { return d.number; }));
    y.domain([0, d3.max(data, function(d) { return d.length; })]);

     course_progress.append("path")
     .attr("transform", 
              "translate(" + margin.left + "," + (margin.top+20) + ")")
        .attr("class", "line")
        .attr("d", valueline(data));

    course_progress.selectAll("dot")
        .data(data)
      .enter().append("circle")
      .attr("transform", 
              "translate(" + margin.left + "," + (margin.top+20) + ")")
        .attr("r", 10)
        .attr("fill","white")
         .attr("cx", function(d) { return x(d.number); })
        .attr("cy", function(d) { return y(d.length); });
        


//Add the course progress bar
course_progress.selectAll("dot")
        .data(data)
      .enter().append('svg:foreignObject')
      .attr("transform", 
              "translate(" + margin.left + "," + (margin.top+20) + ")")
    .attr("width", 120)
    .attr("height", 120)
    .html(function(d) {
        if (d['visited'] == "True") return'<small><i class="fa fa-check-circle fa-2x" style="color:green"></i></small>';
         else if (d['visited'] == "False") return '<small><i class="fa fa-circle fa-2x" style="color:lightgrey"></i></small>';
          else return '<small><i class="fa fa-circle-thin fa-2x" style="color:black"></i></small>';}) // .attr("color",function(d) {if (d['visited'] == "True") return 'green'; else if (d['visited'] == "False") return 'lightgrey'; else return 'black'})
    .attr("x", function(d) { return x(d.number)-10; })
    .attr("y", function(d) { return y(d.length)-12; })
    .on("mouseover", function(d) {
    return tooltip.style("visibility", "visible").text(d['title']);
  })
  .on("mousemove", function() {
    return tooltip.style("top", (d3.event.pageY + 30) + "px")
      .style("left", (d3.event.pageX - 50) + "px");
  })
  
  .on("mouseout", function() {
    return tooltip.style("visibility", "hidden");
  });


//Add the you are here arrow and text
  course_progress.selectAll("dot")
        .data(data)
      .enter().append('foreignObject')
      .attr("transform", 
              "translate(" + (margin.left-60) + "," + (margin.top-15) + ")")
    .attr("width", 120)
    .attr("height", 120)
     .html(function(d) {
        //if (d['here'] == "True") return '<span class="fa-stack fa-3x"><i class="fa fa-caret-down fa-1x"><span class="fa fa-stack-1x" style="color:black;"><span class="here">You are here</span></span></i></span>';
         if (d['here'] == "True") return '<div class=\"stack\"><span class=\"here\">You are here</span><i class=\"fa_custom fa-caret-down fa-2x\"></i></div>';
         else return '';})
    .attr("color",function(d) {if (d['here'] == "True") return 'black'; else return 'white';})
    .attr("x", function(d) { return x(d.number); })
    .attr("y", function(d) { return y(d.length); })
    .on("mouseover", function(d) {
    return tooltip.style("visibility", "visible")
    .text(d['title']);
  })
  
  .on("mousemove", function() {
    return tooltip.style("top", (d3.event.pageY + 30) + "px")
      .style("left", (d3.event.pageX - 50) + "px");
  })
  
  .on("mouseout", function() {
    return tooltip.style("visibility", "hidden");
  });


});


}