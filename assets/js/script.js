
// Get all sections that have an ID defined
const sections = document.querySelectorAll("section[id]");

// Add an event listener listening for scroll
window.addEventListener("scroll", navHighlighter);


function navHighlighter() {

    // Get current scroll position
    let scrollY = window.scrollY;
    let buffer = 150;

    // Now we loop through sections to get height, top and ID values for each
    sections.forEach(current => {
        const sectionHeight = current.offsetHeight;
        const sectionTop = current.offsetTop;
        const sectionId = current.getAttribute("id");

        if (scrollY > sectionTop - buffer && scrollY <= sectionTop + sectionHeight) {
            document.querySelector(".nav a[href*=" + sectionId + "]").classList.add("active");
        } else {
            document.querySelector(".nav a[href*=" + sectionId + "]").classList.remove("active");
        }
    });

}

function createCopyButton(highlightDiv) {
    const button = document.createElement("button");
    button.className = "copy-code-btn text-gray-400 m-0.1 hover:bg-gray-700 rounded-lg py-1 px-1 bg-gray-900 inline-flex items-center justify-center"; // Tailwind CSS classes for styling

    button.innerHTML = '<svg class="w-6 h-6 text-teal-400" viewBox="0 -0.5 25 25" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M17.676 14.248C17.676 15.8651 16.3651 17.176 14.748 17.176H7.428C5.81091 17.176 4.5 15.8651 4.5 14.248V6.928C4.5 5.31091 5.81091 4 7.428 4H14.748C16.3651 4 17.676 5.31091 17.676 6.928V14.248Z" stroke="#a5d6ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M10.252 20H17.572C19.1891 20 20.5 18.689 20.5 17.072V9.75195" stroke="#a5d6ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

    // code copied button
    copy_btn = document.createElement("button");
    copy_btn.className = "fixed z-50 -top-20 left-1/2 transform -translate-x-1/2 p-3 border-0 w-30 h-11 rounded-xl shadow-sm bg-teal-300/80 hover:bg-teal-500 text-slate-900 text-lg font-semibold duration-300 shadow-slate-700 slide-transition transition-all "

    copy_btn.innerHTML = '<div class=" inline-flex"><svg class="h-5 w-5 stroke-slate-900" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <g id="Interface / Check"> <path id="Vector" d="M6 12L10.2426 16.2426L18.727 7.75732"  stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path> </g> </g></svg><span class="text-sm">Code Copied!</span></div>';

    document.body.appendChild(copy_btn);

    button.style.opacity = 1;
    highlightDiv.addEventListener("mouseover", () => {
        button.style.opacity = 1;
    });

    highlightDiv.addEventListener("mouseout", () => {
        button.style.opacity = 1;
    });
    button.addEventListener("click", () => {
        console.log("clicked")
        const codeElement = highlightDiv.querySelector("pre code");
        if (!codeElement) return; // Exit if no code element found
        const textToCopy = codeElement.textContent;
        // const selection = window.getSelection();
        // const range = document.createRange();
        // range.selectNodeContents(codeElement);
        // selection.removeAllRanges();
        // selection.addRange(range);
        navigator.clipboard.writeText(textToCopy).then(() => {
            copy_btn.classList.remove("hidden");
            copy_btn.style.top = "1.5rem";
            setTimeout(() => {
                copy_btn.style.top = "-50px";
            }, 1000); // Remove the notification after 1 seconds
            button.innerHTML = '<svg class="h-6 w-6" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <g id="Interface / Check"> <path id="Vector" d="M6 12L10.2426 16.2426L18.727 7.75732" stroke="#5eead4" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></path> </g> </g></svg>';

        }).catch(err => {
            console.error("Failed to copy code:", err);
            button.innerHTML = '<svg class="w-6 h-6" fill="#e60000" viewBox="0 0 200 200" data-name="Layer 1" id="Layer_1" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"><title></title><path d="M114,100l49-49a9.9,9.9,0,0,0-14-14L100,86,51,37A9.9,9.9,0,0,0,37,51l49,49L37,149a9.9,9.9,0,0,0,14,14l49-49,49,49a9.9,9.9,0,0,0,14-14Z"></path></g></svg>';
        });

        setTimeout(() => {
            button.innerHTML = '<svg class="w-6 h-6 text-teal-400" viewBox="0 -0.5 25 25" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M17.676 14.248C17.676 15.8651 16.3651 17.176 14.748 17.176H7.428C5.81091 17.176 4.5 15.8651 4.5 14.248V6.928C4.5 5.31091 5.81091 4 7.428 4H14.748C16.3651 4 17.676 5.31091 17.676 6.928V14.248Z" stroke="#a5d6ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M10.252 20H17.572C19.1891 20 20.5 18.689 20.5 17.072V9.75195" stroke="#a5d6ff" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
        }, 1000); // Change button text back to "Copy" after 1 second
    });



    highlightDiv.appendChild(button);
}

function createQuoteIcon(blockquote) {
    const currentHTML = blockquote.innerHTML
    blockquote.innerHTML = '<svg class="w-10 h-10 mx-auto mb-3 text-gray-600" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 14"><path d="M6 0H2a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h4v1a3 3 0 0 1-3 3H2a1 1 0 0 0 0 2h1a5.006 5.006 0 0 0 5-5V2a2 2 0 0 0-2-2Zm10 0h-4a2 2 0 0 0-2 2v4a2 2 0 0 0 2 2h4v1a3 3 0 0 1-3 3h-1a1 1 0 0 0 0 2h1a5.006 5.006 0 0 0 5-5V2a2 2 0 0 0-2-2Z"/></svg>' + currentHTML;
    console.log("HELLO")
}

document.querySelectorAll(".highlight").forEach(createCopyButton);
document.querySelectorAll("blockquote").forEach(createQuoteIcon);


$(document).ready(function () {
    $("#TableOfContents").addClass("nav hidden lg:block");
    $("nav#TableOfContents ul").addClass("mt-16 w-max");
    $("nav#TableOfContents ul li a").addClass("group flex items-center py-3 active")
});



let lastScrollTop = 0;
const button = document.getElementById('to-top-button');
const scrollThreshold = window.innerHeight - 200; // Set the scroll threshold

window.addEventListener('scroll', function () {
    let scrollTop = window.scrollY || document.documentElement.scrollTop;
    if (scrollTop > scrollThreshold && scrollTop < lastScrollTop) {
        // Scrolling up
        button.classList.remove('hidden');
        button.style.top = "1.25rem";
    } else {
        // Scrolling down
        button.style.top = "-50px";

    }

    lastScrollTop = scrollTop <= 0 ? 0 : scrollTop; // For Mobile or negative scrolling
});

// Smooth scroll to top
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}


const elPointer = document.querySelector("#glow_box");

addEventListener("mousemove", (evt) => {
    background = `radial-gradient(600px at ${$(document).scrollLeft() + evt.clientX}px ${$(document).scrollTop() + evt.clientY}px, rgba(29, 78, 216, 0.20), transparent 80%)`
    elPointer.style.setProperty("--bg-gradient", background);
});


window.addEventListener("scroll", function () {
    const scrollableHeight =
        document.documentElement.scrollHeight - window.innerHeight;
    const scrolled = window.scrollY;
    const progressBar = document.getElementById("scroll-progress");
    const progress = (scrolled / scrollableHeight) * 100;
    progressBar.style.width = progress + "%";
});

//  KATEX

document.addEventListener("DOMContentLoaded", function () {
    renderMathInElement(document.body, {
        delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false }
        ]
    });
});