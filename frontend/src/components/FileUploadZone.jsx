import { useDropzone } from "react-dropzone";

export default function FileUploadZone({ onFileSelected, disabled = false }) {
  const { getRootProps, getInputProps, acceptedFiles } = useDropzone({
    multiple: false,
    disabled,
    accept: {
      "text/csv": [".csv"],
      "application/json": [".json"],
      "application/pdf": [".pdf"],
    },
    onDrop: (files) => {
      if (files?.[0]) {
        onFileSelected(files[0]);
      }
    },
  });

  return (
    <div {...getRootProps()} className="dropzone">
      <input {...getInputProps()} />
      <p>Drag and drop a CSV, JSON, or PDF invoice file here, or click to choose one.</p>
      {acceptedFiles[0] ? <p className="muted">Selected: {acceptedFiles[0].name}</p> : null}
    </div>
  );
}
