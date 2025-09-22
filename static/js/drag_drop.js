const dropArea = document.getElementById('drop-area');
const dropArea2 = document.getElementById('drop-area2');

const inputFile = document.getElementById('input-file');
const inputFile2 = document.getElementById('input-file2');

const imageView = document.getElementById('img-view');
const imageView2 = document.getElementById('img-view2');

inputFile.addEventListener("change", uploadImage);

function uploadImage(){
    let imgLink = URL.createObjectURL(inputFile.files[0]);
    imageView.style.backgroundImage = `url(${imgLink})`;
    imageView.textContent = '';
    imageView.style.border = 0;
}

dropArea.addEventListener("dragover", function(e){
    e.preventDefault();
});
dropArea.addEventListener("drop", function(e){
    e.preventDefault();
    inputFile.files = e.dataTransfer.files;
    uploadImage();
});
////////////////////////////////////////////////////
inputFile2.addEventListener("change", uploadImage2);

function uploadImage2(){
    let imgLink = URL.createObjectURL(inputFile2.files[0]);
    imageView2.style.backgroundImage = `url(${imgLink})`;
    imageView2.textContent = '';
    imageView2.style.border = 0;
}

dropArea2.addEventListener("dragover", function(e){
    e.preventDefault();
});
dropArea2.addEventListener("drop", function(e){
    e.preventDefault();
    inputFile2.files = e.dataTransfer.files;
    uploadImage2();
});