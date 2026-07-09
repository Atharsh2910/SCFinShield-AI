import { useDropzone } from "react-dropzone";

export default function FileUploadZone({ onFileSelected, disabled = false }) {
  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    multiple: false,
    disabled,
    accept: {
      "text/csv": [".csv"],
      "application/json": [".json"],
      "application/pdf": [".pdf"],
    },
    onDrop: (files) => {
      if (files?.[0]) onFileSelected(files[0]);
    },
  });

  const file = acceptedFiles[0];

  return (
    <div
      {...getRootProps()}
      className="dropzone"
      id="invoice-file-dropzone"
      style={
        isDragActive
          ? { borderColor: "#06b6d4", background: "rgba(6,182,212,0.08)" }
          : undefined
      }
    >
      <input {...getInputProps()} id="invoice-file-input" />
      <div style={{ fontSize: 36, marginBottom: 12 }}>📂</div>
      {isDragActive ? (
        <p style={{ color: "#06b6d4", fontWeight: 600 }}>Drop file to upload…</p>
      ) : (
        <p>
          Drag &amp; drop a <strong>CSV</strong>, <strong>JSON</strong>, or{" "}
          <strong>PDF</strong> invoice file here, or{" "}
          <span style={{ color: "#06b6d4", cursor: "pointer" }}>browse</span>
        </p>
      )}
      {file ? (
        <p className="muted" style={{ marginTop: 8 }}>
          ✅ Selected: <strong>{file.name}</strong> (
          {(file.size / 1024).toFixed(1)} KB)
        </p>
      ) : null}
    </div>
  );
}
