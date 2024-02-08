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
            var reader = new FileReader();
            reader.onload = function(event) {
                var imgTmp = new Image();
                imgTmp.onload = function() {
                    imageUploadedDisplayed.getContext('2d').drawImage(imgTmp, 0, 0);

                    setTimeout(function() {
                        var ctxResult = imageResult.getContext('2d');
                        ctxResult.clearRect(0, 0, imageResult.width, imageResult.height);
                        ctxResult.drawImage(imgTmp, 0, 0); // Draw the original image first
                        var imageData = ctxResult.getImageData(0, 0, imageResult.width, imageResult.height);
                        var data = imageData.data;
                        for (var i = 0; i < data.length; i += 4) {
                            var avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
                            // RBG
                            data[i] = avg;
                            data[i + 1] = avg;
                            data[i + 2] = avg;
                        }
                        ctxResult.putImageData(imageData, 0, 0);
                    }, 3000);
                };
                imgTmp.src = event.target.result;
            };
            reader.readAsDataURL(targetImage);
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

document.querySelector('form').addEventListener('submit', function(event)
{
    event.preventDefault();

    var imageResult = document.getElementById("imageResult");
    var imageUploadedDisplayed = document.getElementById("imageUploadedDisplayed");
    var formImage = document.getElementById("imageUploaded").files[0];
    var formData = new FormData();
    formData.append("imageUploadedDisplayed",formImage);
    var xhr = new XMLHttpRequest();

    xhr.open('POST', '/', true);
    xhr.onload = function() {
        if (xhr.status === 200) {
            var blob=new Blob([xhr.response],{type: 'image/jpeg'});
            var imageUrl=URL.createObjectURL(blob)
            var tmp=new Image();

            tmp.onload = function() {
                imageResult.width = tmp.width;
                imageResult.height = tmp.height;
                imageResult.getContext('2d').drawImage(tmp,0,0);
                URL.revokeObjectURL(imageUrl);
            };
            tmp.src = imageUrl;

            console.log('Request successful');
            console.log(imageUrl)
            console.log(imageResult.width)
            console.log(imageResult.height)
        } else {
            console.log('Request failed');
        }
    };

    xhr.send(formData);
});