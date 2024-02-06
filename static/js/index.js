function isCanvasBlank(canvas) {
  const context = canvas.getContext('2d');

  const pixelBuffer = new Uint32Array(
    context.getImageData(0, 0, canvas.width, canvas.height).data.buffer
  );

  return !pixelBuffer.some(color => color !== 0);
}

function cancelAction(){
    var imageUploadedDisplayed = document.getElementById("imageUploadedDisplayed");
    var imageUploaded = document.getElementById("imageResult");

    if(isCanvasBlank(imageUploadedDisplayed))
    {
        alertHelper.alert("Canvas is empty");
    }
    else {
        var imageUploadedDisplayedContext = imageUploadedDisplayed.getContext('2d');
        imageUploadedDisplayedContext.clearRect(0, 0, imageUploadedDisplayed.width, imageUploadedDisplayed.height)

        var imageUploadedContext = imageUploaded.getContext('2d');
        imageUploadedContext.clearRect(0, 0, imageUploaded.width, imageUploaded.height)

        alertHelper.alert('Image discarded')

        setTimeout(AlertHelper,200);
    }
}

function saveImage(){
    var imageUploadedDisplayed = document.getElementById("imageUploadedDisplayed");
    var imageUploaded = document.getElementById("imageResult").toDataURL("image/png");

    if(isCanvasBlank(imageUploadedDisplayed)) {
        alertHelper.alert("Canvas is empty");
    }
    else{
        let savingDetails = document.createElement("a");
        savingDetails.href=imageUploaded;
        savingDetails.download="depixelated_image.png";
        savingDetails.click();
        alertHelper.alert("Image saved");
    }
}

function uploadImage(){
    var imageResult = document.getElementById("imageResult");
    var imageUploadedDisplayed = document.getElementById("imageUploadedDisplayed");
    var imageUploaded = document.getElementById("imageUploaded")

    const listImages=imageUploaded.files;
    if(listImages.length>1){
        alertHelper.alert("Too many files selected. Images should be added individually");
    }
    else {
        const targetImage = listImages[0];
        if (!targetImage) {
            // nothing has been uploaded yet
            return;
        }
        const filename = targetImage.name;
        const extension = filename.split('.').pop().toLowerCase();
        const extensionsAllowed = ["jpg", "jpeg"];

        if (extensionsAllowed.includes(extension) === false)
            alertHelper.alert("File extension '." + extension + "' is not supported. " +
                "Supported types are: \'.jpg\', \'.jpeg\', \'.JPG\', \'.JPEG\'. ");
        else {
            var imgTmp = new SimpleImage(imageUploaded);

            imgTmp.drawTo(imageResult);
            imgTmp.drawTo(imageUploadedDisplayed)
        }
    }
}

function AlertHelper(){
  this.alert = function(message,title){
    document.body.innerHTML = document.body.innerHTML +
        '<div id="dialogoverlay"></div>' +
        '<div id="dialogbox"><div>' +
        '<div id="dialogboxbody"></div>' +
        '<div id="dialogboxfoot"></div></div></div>';

    let dialogoverlay = document.getElementById('dialogoverlay');
    let dialogbox = document.getElementById('dialogbox');

    let winH = window.innerHeight;
    dialogoverlay.style.height = winH+"px";

    dialogbox.style.top = "100px";

    dialogoverlay.style.display = "block";
    dialogbox.style.display = "block";

    document.getElementById('dialogboxbody').innerHTML = message;
    document.getElementById('dialogboxfoot').innerHTML =
        '<button onclick="alertHelper.ok()">OK</button>';
  }

  this.ok = function(){
    document.getElementById('dialogbox').style.display = "none";
    document.getElementById('dialogoverlay').style.display = "none";
  }
}

let alertHelper = new AlertHelper();