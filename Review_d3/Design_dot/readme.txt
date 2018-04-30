To use this:
1. Change the references for the javascript, csv, json and css files to wherever the file is located. Best to use a https location. Currently it is uploaded to a ucb server and loaded from the server.
2. Open the html page in the browser.
3. There are 2 javascript files:
	radial.js - which renders the radial chart, links and skill coverage bar
	d3Progress.js - which renders the course progress bar
4. percent.csv - which has skill,score,top_score,covered; where skill is the skill id, scores are the output from the model, covered depends on the navigation of the course specific to a skill
5. modules.json - json file containing the course information like chapters, sections and sub-sections, title and if a student has visited those links and if the student is currently at that location.
6. skill_link.json - which contains the mapping of skill id to a skill name and the associated links which would be outputted by the model as suggested resources to review.