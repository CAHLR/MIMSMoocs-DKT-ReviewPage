/* Reusable, pure d3 Checkbox */
/* -- Reference https://bl.ocks.org/Lulkafe/c77a36d5efb603e788b03eb749a4a714 */

function d3CheckBox () {

    var size = 19,
        x = 0,
        y = 0,
        rx = 0,
        ry = 0,
        markStrokeWidth = 3,
        boxStrokeWidth = 3,
        checked = true;

    function checkBox (selection) {

        var g = selection.append("g"),
            box = g.append("rect")
            .attr("width", size)
            .attr("height", size)
            .attr("x", x)
            .attr("y", y)
            .attr("rx", rx)
            .attr("ry", ry)
            .style({
                "fill-opacity": 0,
                "stroke-width": boxStrokeWidth,
                "stroke": "black",
                "visibility":"hidden"
            });

        //Data to represent the check mark
        var coordinates = [
            {x: x + (size / 4)-(size/Math.sqrt(2)), y: y + (size / 1.6)-(size/Math.sqrt(2))},
            {x: x + (size / 1.8)-(size/Math.sqrt(2)), y: (y + size) - (size / 10)+ (size / 20)-(size/Math.sqrt(2))},
            {x: (x + size) - (size / 100)-(size/Math.sqrt(2))+ (size*0.1), y: (y + (size / 2.6)-(size/Math.sqrt(2)))}
        ];

        var line = d3.svg.line()
                .x(function(d){ return d.x; })
                .y(function(d){ return d.y; })
                .interpolate("basic");

        var mark = g.append("path")
            .attr("d", line(coordinates))
            .style({
                "stroke-width" : markStrokeWidth,
                "stroke" : "white",
                "fill" : "none",
                "opacity": (checked)? 1 : 0
            });

        // g.on("click", function () {
        //     checked = !checked;
        //     mark.style("opacity", (checked)? 1 : 0);

        //     if(clickEvent)
        //         clickEvent();

        //     d3.event.stopPropagation();
        // });

    }

    checkBox.size = function (val) {
        size = val;
        return checkBox;
    }

    checkBox.x = function (val) {
        x = val;
        return checkBox;
    }


    checkBox.y = function (val) {
        y = val;
        return checkBox;
    }

    checkBox.rx = function (val) {
        rx = val;
        return checkBox;
    }

    checkBox.ry = function (val) {
        ry = val;
        return checkBox;
    }

    checkBox.markStrokeWidth = function (val) {
        markStrokeWidth = val;
        return checkBox;
    }

    checkBox.boxStrokeWidth = function (val) {
        boxStrokeWidth = val;
        return checkBox;
    }

    checkBox.checked = function (val) {

        if(val === undefined) {
            return checked;
        } else {
            checked = val;
            return checkBox;
        }
    }

    checkBox.clickEvent = function (val) {
        clickEvent = val;
        return checkBox;
    }

    return checkBox;
}