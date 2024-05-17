function previousNextImage(n)
{
  showSlides(imageIndex += n);
}

function showSlides(n)
{
  let processedImages = document.getElementsByClassName("processedImages");
  let dots = document.getElementsByClassName("dot");
  if (n > processedImages.length) {imageIndex = 1}
  if (n < 1) {imageIndex = processedImages.length}
  for (let i = 0; i < processedImages.length; i++) {
    processedImages[i].style.display = "none";
  }
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active", "");
  }
  processedImages[imageIndex-1].style.display = "block";
  dots[imageIndex-1].className += " active";
}

let imageIndex = 1;
showSlides(imageIndex);


// window.onload=function(){
//     var message = "{{ get_flashed_messages() [0] if get_flashed_messages() else '' }}";
//     if(message) {
//         window.alert(message)
//     }
// }