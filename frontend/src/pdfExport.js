import jsPDF from "jspdf";

export function exportResultToPDF(result, gapReport) {
  const doc = new jsPDF();
  let y = 20;

  doc.setFontSize(18);
  doc.text("CareerLens Report", 14, y);
  y += 12;

  doc.setFontSize(12);
  doc.text(`Top predicted career: ${result.top_prediction}`, 14, y);
  y += 8;

  if (result.top_5?.length) {
    doc.setFontSize(11);
    doc.text("Top 5 matches (trait-based):", 14, y);
    y += 7;
    result.top_5.forEach((r) => {
      doc.text(`  ${r.career} — ${Math.round(r.confidence * 100)}%`, 14, y);
      y += 6;
    });
    y += 4;
  }

  if (result.tech_top_5?.length) {
    doc.text("Top 5 matches (tech-role, resume-based):", 14, y);
    y += 7;
    result.tech_top_5.forEach((r) => {
      doc.text(`  ${r.career} — ${Math.round(r.confidence * 100)}%`, 14, y);
      y += 6;
    });
    y += 4;
  }

  if (gapReport) {
    doc.setFontSize(13);
    doc.text(`Skill Gap Report — ${gapReport.career}`, 14, y);
    y += 8;
    doc.setFontSize(11);
    doc.text(`Overall readiness: ${gapReport.overall_readiness}%`, 14, y);
    y += 8;
    gapReport.gaps.forEach((g) => {
      if (y > 270) { doc.addPage(); y = 20; }
      doc.text(`  ${g.trait}: ${g.resume_score}/${g.target_score} (${g.severity})`, 14, y);
      y += 6;
    });
  }

  doc.save("careerlens-report.pdf");
}
