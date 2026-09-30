export default function Toasts({ toasts }) {
  return (
    <div className="toasts" aria-live="polite">
      {toasts.map((t) => <p key={t.id} className={`toast toast-${t.tone}`}>{t.text}</p>)}
    </div>
  );
}
