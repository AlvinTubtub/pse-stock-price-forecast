"use client";

import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "Home", icon: "🏠" },
  { href: "/companies", label: "Companies", icon: "🏢" },
  { href: "/compare", label: "Models", icon: "📊" },
  { href: "/learn", label: "Learn", icon: "📖" },
  { href: "/about", label: "About", icon: "ℹ️" },
];

export default function MobileNav() {
  const pathname = usePathname();

  return (
    <nav className="fixed bottom-0 w-full glass z-50 border-t border-dark-border md:hidden">
      <div className="flex items-center justify-around h-16">
        {LINKS.map((link) => {
          const active =
            link.href === "/"
              ? pathname === "/"
              : pathname.startsWith(link.href);

          return (
            <a
              key={link.href}
              href={link.href}
              className={`flex flex-col items-center gap-0.5 text-[11px] px-2 py-1 rounded-md ${
                active ? "text-brand-400" : "text-slate-400"
              }`}
            >
              <span className="text-lg leading-none">
                {link.icon}
              </span>

              {link.label}
            </a>
          );
        })}
      </div>
    </nav>
  );
}