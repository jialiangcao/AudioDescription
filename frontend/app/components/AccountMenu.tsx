"use client";

import { useEffect, useRef, useState } from "react";
import { signOut, useAccount } from "./AuthGate";

/**
 * The account control: an avatar button that opens the signed-in address and
 * the way out. Both used to sit in the header in plain sight; neither is
 * something anyone needs while watching a video.
 */
export default function AccountMenu() {
  const { email } = useAccount();
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  // Close on an outside click or Escape, the way any menu is expected to.
  useEffect(() => {
    if (!open) return;
    const onPointer = (e: MouseEvent) => {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // With auth switched off there is no account to show, but the header still
  // wants something in that slot, so keep the button and say as much.
  const initial = (email ?? "?").trim().charAt(0).toUpperCase();

  return (
    <div className="account" ref={wrapRef}>
      <button
        type="button"
        className="avatar"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label="Account"
        onClick={() => setOpen((prev) => !prev)}
      >
        {initial}
      </button>
      {open && (
        <div className="account-menu" role="menu">
          <p className="account-email">{email ?? "Signed in locally"}</p>
          {email && (
            <button
              type="button"
              className="account-signout"
              role="menuitem"
              onClick={() => {
                setOpen(false);
                void signOut();
              }}
            >
              Sign out
            </button>
          )}
        </div>
      )}
    </div>
  );
}
