window.onload=function(){
    var message = "{{ get_flashed_messages() [0] if get_flashed_messages() else '' }}";
    if(message) {
        window.alert(message)
    }
}