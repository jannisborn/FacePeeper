      var spinnWheelOpts = {
          lines: 11 // The number of lines to draw
        , length: 28 // The length of each line
        , width: 12 // The line thickness
        , radius: 56 // The radius of the inner circle
        , scale: 0.5 // Scales overall size of the spinner
        , corners: 1 // Corner roundness (0..1)
        , color: '#000' // #rgb or #rrggbb or array of colors
        , opacity: 0 // Opacity of the lines
        , rotate: 0 // The rotation offset
        , direction: 1 // 1: clockwise, -1: counterclockwise
        , speed: 0.7 // Rounds per second
        , trail: 100 // Afterglow percentage
        , fps: 20 // Frames per second when using setTimeout() as a fallback for CSS
        , zIndex: 2e9 // The z-index (defaults to 2000000000)
        , className: 'spinner' // The CSS class to assign to the spinner
        , top: '50%' // Top position relative to parent
        , left: '50%' // Left position relative to parent
        , shadow: false // Whether to render a shadow
        , hwaccel: false // Whether to use hardware acceleration
        , position: 'relative' // Element positioning
    };
    var globalActorList = [];
    var spinner = new Spinner(spinnWheelOpts);


    $(document).ready(function(){

        //set up the drag and drop zone
        var dropZone = document.getElementById('dropzone');
        dropZone.addEventListener('dragover', handleDragOver, false);
        dropZone.addEventListener('drop', handleFileSelect, false);

        //set up the little pen button
        $(".icon#Edit").click(enterCorrectionMode);
        //set up the send correction button
        $("#correction_selector button").click(sendCorrection);


    // load the list of all our actors from the server
    // put all of them in the options list that the user has to select from when 
    // he wants to correct the classification
    dataList = document.getElementById("actors_list");
    $.get("api/actorList", function(actorList){
        globalActorList = actorList;
        //save the list so we can later use it to check for correct input
         actorList.forEach(function(actor) {
            // Create a new <option> element.
            var option = document.createElement('option');
            // Set the value using the item in the JSON array.
            option.value = actor;
            // Add the <option> element to the <datalist>.
            dataList.appendChild(option);
        });
    }, "json");

});


function enterCorrectionMode(){
    $(".normal_title").hide();
    $("#icon_buttons").hide();
    $("#correction_selector").show();
}

function sendCorrection(){
        var myNewName = $("#actor_options").val();
        if(-1 != $.inArray(myNewName, globalActorList)){ 
            $.post("/api/correctClassification/"+sessionStorage.currentImageId,
                {newName: myNewName},
                function(data,status){
                    if(status!="success"){
                        alert("We couldn't update the classifier: " + data.message + " status: "+ status)
                    }
                });
            // we display the info, but do not allow correction
            displayActorInfo(myNewName, false);
        }
        else{
            var inputfield = $("#correction_selector");
            // wonderfull shaky animation for incorrect user input
            inputfield.animate({paddingLeft: "10px"}, 70);
            inputfield.animate({paddingLeft: "0px"}, 60);
            inputfield.animate({paddingLeft: "10px"}, 50);
            inputfield.animate({paddingLeft: "0px"}, 50);
            inputfield.animate({paddingLeft: "10px"}, 60);
            inputfield.animate({paddingLeft: "0px"}, 70);
           // inputfield.animate({paddingLeft: "10px"}, 250);
        }
        $("#actor_options").val("");
    }

    function displayActorInfo(name, allow_edit){
        /**displays name and info for a given actor. might be from server or from user correction*/
        $.get("/api/actorInfo/" + name,
                function(data, status){
                    if(status=="success"){
                        $("#actor_text").text(data);
                    }
                    else{
                        $("#actor_text").text("Server didn't give us data" + status);
                    }
            });
        $("#actor_name").text(name);
        $("#correction_selector").hide();
        $(".normal_display").show();
        if(allow_edit){
            $("#icon_buttons").show();
        }
    }

    function handleDragOver(evt) {
        evt.stopPropagation();
        evt.preventDefault();
        evt.dataTransfer.dropEffect = 'copy'; // Explicitly show this is a copy.
    }

    function handleFileSelect(evt) {
        evt.stopPropagation();
        evt.preventDefault();


        var file = evt.dataTransfer.files[0];
        //check if it is image
        if(null != file.type.match("image/*")){
            var target = document.getElementById('dropzone');
            document.getElementById('help_text').style.display= "none";
            spinner.spin(target);
            uploadFile(file);
        }
        else{
            alert("please choose an image file!");
        }
    }

    function uploadFile(file){

        var formdata = new FormData();
        formdata.append("file", file)
        var xhttp = new XMLHttpRequest();

        xhttp.onreadystatechange = function(){         
            if (this.readyState == 4 && this.status == 200) {
                let respObject = jQuery.parseJSON(this.responseText);
                processClassificationResult(respObject);
            }
            else if(this.readyState == 4 && this.status == 400){
		var serverMessage = jQuery.parseJSON(this.responseText).message;
                alert("The server could not classify your image: " + serverMessage);
                location.reload();
            }
            else if(this.readyState == 4){
                alert("Unknown server error");
                location.reload();
	    }
	
        };

        sessionStorage.currentImageId =new Date().getTime()%1000000
        xhttp.open("POST", "api/classifyImage/"+sessionStorage.currentImageId, true);
        xhttp.send(formdata);
        return false;
    }

    function processClassificationResult(answer){
        // we classified an image. we have to do the following
        // load the cropped preview face
        var imgSrc = "/api/getPreProcessedImg/" + sessionStorage.currentImageId;
        // once it is loaded, we display it together with the actor info
        $("#preview_img").one("load", function(){displayClassificationResult(answer);});
        $("#preview_img").attr("src", imgSrc);
        $("#preview_img").show();
    }

    function displayClassificationResult(answer){
        /**display a new server classification*/
        spinner.stop();
        var name = answer.label;
        // we display the info and allow editing
        displayActorInfo(name, true)
    }

