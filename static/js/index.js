function saveImage(){
    alert("Image saved");
}

function isCanvasBlank(canvas) {
  const context = canvas.getContext('2d');

  const pixelBuffer = new Uint32Array(
    context.getImageData(0, 0, canvas.width, canvas.height).data.buffer
  );

  return !pixelBuffer.some(color => color !== 0);
}

// function cancelActionImageDiscardedConfirmation()
// {
//     cancelAction();
//     setTimeout(canvasDeleted,1000);
// }

function cancelAction(){
    var imageUploadedDisplayed = document.getElementById("imageUploadedDisplayed");
    var imageUploaded = document.getElementById("imageResult");

    if(isCanvasBlank(imageUploadedDisplayed))
    {
        alert("Canvas already empty");
    }
    else {
        var imageUploadedDisplayedContext = imageUploadedDisplayed.getContext('2d');
        imageUploadedDisplayedContext.clearRect(0, 0, imageUploadedDisplayed.width, imageUploadedDisplayed.height)

        var imageUploadedContext = imageUploaded.getContext('2d');
        imageUploadedContext.clearRect(0, 0, imageUploaded.width, imageUploaded.height)

        setTimeout(canvasDeleted,500);
    }
}

function canvasDeleted()
{
    alert("Image discarded");
}

function uploadImage(){
    var imageResult = document.getElementById("imageResult");
    var imageUploadedDisplayed = document.getElementById("imageUploadedDisplayed");
    var imageUploaded = document.getElementById("imageUploaded")
    var imgTmp = new SimpleImage(imageUploaded);

    imgTmp.drawTo(imageResult);
    imgTmp.drawTo(imageUploadedDisplayed)
}