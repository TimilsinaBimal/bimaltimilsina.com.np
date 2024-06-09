import { fontFamily as _fontFamily } from "tailwindcss/defaultTheme";

export const content = ["./layouts/*.html", "./layouts/**/*.html", "./content/*/*/*.md", "./assets/js/*.js"];
export const theme = {
    extend: {
        fontFamily: {
            sans: ["Geist", "Inter", ..._fontFamily.sans],
            mono: ["Geist Mono", ..._fontFamily.mono],
        }
    },
};
export const plugins = [require("@tailwindcss/typography")];