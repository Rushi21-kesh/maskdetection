













/**************** Scroll Indicator ******************/

let progressSection = document.querySelector('.progress-section');
let progressBar = document.querySelector('.progress-bar');
let progressNum = document.querySelector('.progress-num');



function updateProgressBar()
{

    progressBar.style.height = `${getScrollPercentage()}%`
  
    requestAnimationFrame(updateProgressBar)
}

function getScrollPercentage()
{
    return (100-window.scrollY/(document.body.scrollHeight - window.innerHeight)*100);
}

updateProgressBar()


/*************** Hamburger Toggle****************/

const hamburger = document.querySelector(".hamburger");
const navMenu = document.querySelector(".nav-list");

hamburger.addEventListener("click", mobileMenu);

function mobileMenu() {
    hamburger.classList.toggle("active");
    navMenu.classList.toggle("active");
}