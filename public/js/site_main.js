
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

        if (sectionId == "about") {
            console.log(sectionHeight)
        }
        if (sectionId == "experience") {
            console.log(`scroll: ${scrollY}, Top ${sectionTop}, ${sectionHeight}`);
        }
        if (scrollY > sectionTop - buffer && scrollY <= sectionTop + sectionHeight) {
            document.querySelector(".nav a[href*=" + sectionId + "]").classList.add("active");
        } else {
            document.querySelector(".nav a[href*=" + sectionId + "]").classList.remove("active");
        }
    });

}

function createCopyButton(highlightDiv) {
    const button = document.createElement("button");
    button.className = "copy-code-btn text-gray-500 dark:text-gray-400 m-0.2 hover:bg-gray-100 dark:bg-gray-800 dark:border-gray-600 dark:hover:bg-gray-700 rounded-lg py-2 px-2.5 inline-flex items-center justify-center bg-white border-gray-200 border"; // Tailwind CSS classes for styling

    button.innerHTML = '<svg class="w-3.5 h-3.5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 20"><path d="M16 1h-3.278A1.992 1.992 0 0 0 11 0H7a1.993 1.993 0 0 0-1.722 1H2a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2Zm-3 14H5a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2Zm0-4H5a1 1 0 0 1 0-2h8a1 1 0 1 1 0 2Zm0-5H5a1 1 0 0 1 0-2h2V2h4v2h2a1 1 0 1 1 0 2Z"/></svg>';


    button.addEventListener("click", () => {
        const codeElement = highlightDiv.querySelector("pre code");
        if (!codeElement) return; // Exit if no code element found
        const textToCopy = codeElement.textContent;
        // const selection = window.getSelection();
        // const range = document.createRange();
        // range.selectNodeContents(codeElement);
        // selection.removeAllRanges();
        // selection.addRange(range);
        navigator.clipboard.writeText(textToCopy).then(() => {
            button.innerHTML = '<svg class="w-3.5 h-3.5 text-teal-700 dark:text-teal-500" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 16 12"><path stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M1 5.917 5.724 10.5 15 1.5"/></svg>';

        }).catch(err => {
            console.error("Failed to copy code:", err);
            button.textContent = "Copy Failed";
        });

        setTimeout(() => {
            button.innerHTML = '<svg class="w-3.5 h-3.5" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="currentColor" viewBox="0 0 18 20"><path d="M16 1h-3.278A1.992 1.992 0 0 0 11 0H7a1.993 1.993 0 0 0-1.722 1H2a2 2 0 0 0-2 2v15a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V3a2 2 0 0 0-2-2Zm-3 14H5a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2Zm0-4H5a1 1 0 0 1 0-2h8a1 1 0 1 1 0 2Zm0-5H5a1 1 0 0 1 0-2h2V2h4v2h2a1 1 0 1 1 0 2Z"/></svg>';
        }, 1000); // Change button text back to "Copy" after 1 second
    });

    highlightDiv.appendChild(button);
}

document.querySelectorAll(".highlight").forEach(createCopyButton);

$(document).ready(function () {
    $("#TableOfContents").addClass("nav hidden lg:block");
    $("nav#TableOfContents ul").addClass("mt-16 w-max");
    $("nav#TableOfContents ul li a").addClass("group flex items-center py-3 active")
});


const scrollToTopButton =
    document.getElementById('to-top-button');

// Show button when user scrolls down
window.addEventListener('scroll', () => {
    if ($(document).scrollTop() > 500) {
        scrollToTopButton.style.display = 'block';
    } else {
        scrollToTopButton.style.display = 'none';
    }
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
