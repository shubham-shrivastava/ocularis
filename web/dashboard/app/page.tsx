import { RunList } from "@/components/RunList";

export default function Home() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-white">Runs</h1>
        <p className="text-sm text-zinc-500 mt-1">
          All agent runs. Click a run to see live view and controls.
        </p>
      </div>
      <RunList />
    </div>
  );
}
