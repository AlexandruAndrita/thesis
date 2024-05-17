function previousNextImage(n)
{
  showSlides(imageIndex += n);
}

function confirmDiscardImages(endpoint) {
    console.log(endpoint);
    let discardImageConfiguration = {
        title: 'Are you sure?',
        icon: 'warning',
        allowOutsideClick: false,
        allowEscapeKey: false,
        showCancelButton: true,
        confirmButtonColor: '#0000e6',
        cancelButtonColor: '#e60000',
        confirmButtonText: 'Yes',
        cancelButtonText: 'No'
    };
    let errorConfiguration={
        title : 'Error',
        icon: 'error',
        allowOutsideClick: false,
        allowEscapeKey: false,
        showCancelButton: true,
        confirmButtonColor: '#0000e6',
    };
    Swal.fire(discardImageConfiguration).then((result) => {
        if (result.isConfirmed) {
            fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({})
            })
            .then(response => {
                if (response.redirected) {
                    window.location.href = response.url;
                } else {
                    return response.json();
                }
            })
            .catch(error => {
                errorConfiguration.text = error;
                Swal.fire(errorConfiguration);
            });
        }
    });
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
