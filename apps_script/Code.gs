/**
 * Google Apps Script Web App Bridge for the Financial ML Preprocessing Pipeline.
 *
 * Allows GitHub Actions (or local runs) to read the source Google Sheet and publish
 * formatted processed tabs to the destination Google Sheet without Google Cloud Console.
 *
 * Setup:
 *   1. Open your Destination Google Sheet -> Extensions -> Apps Script.
 *   2. Replace Code.gs contents with this script and set API_TOKEN below.
 *   3. Click Deploy -> New deployment -> Web app
 *      - Execute as: Me
 *      - Who has access: Anyone
 *   4. Authorize when prompted and copy the /exec Web App URL.
 */

const API_TOKEN = "REPLACE_WITH_YOUR_SECRET_TOKEN";

function jsonResponse(obj) {
  return ContentService
    .createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}

function doGet() {
  return jsonResponse({ ok: true, status: "Financial ML Pipeline Apps Script bridge is active." });
}

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return jsonResponse({ ok: false, error: "Empty POST body." });
    }
    const req = JSON.parse(e.postData.contents);
    if (!API_TOKEN || API_TOKEN === "REPLACE_WITH_YOUR_SECRET_TOKEN" || req.token !== API_TOKEN) {
      return jsonResponse({ ok: false, error: "Unauthorized token." });
    }

    const action = req.action;
    if (action === "fetch") {
      return handleFetch(req);
    }
    if (action === "get_tail") {
      return handleGetTail(req);
    }
    if (action === "append_row") {
      return handleAppendRow(req);
    }
    if (action === "prepare_tab") {
      return handlePrepareTab(req);
    }
    if (action === "write_chunk") {
      return handleWriteChunk(req);
    }
    if (action === "format_tab") {
      return handleFormatTab(req);
    }
    if (action === "apply_color_scales") {
      return handleApplyColorScales(req);
    }
    if (action === "rebuild_charts") {
      return handleRebuildCharts(req);
    }
    if (action === "publish_doc") {
      return handlePublishDoc(req);
    }
    return jsonResponse({ ok: false, error: "Unknown action: " + action });
  } catch (err) {
    return jsonResponse({ ok: false, error: String(err && err.stack ? err.stack : err) });
  }
}

function handleGetTail(req) {
  const ss = SpreadsheetApp.openById(req.source_id);
  const sheet = req.sheet_name ? ss.getSheetByName(req.sheet_name) : ss.getSheets()[0];
  if (!sheet) {
    return jsonResponse({ ok: false, error: "Sheet not found: " + req.sheet_name });
  }
  const lastRow = sheet.getLastRow();
  const lastCol = sheet.getLastColumn();
  if (lastRow === 0 || lastCol === 0) {
    return jsonResponse({ ok: true, headers: [], tail_rows: [], last_row: 0 });
  }
  const headers = sheet.getRange(1, 1, 1, lastCol).getDisplayValues()[0];
  const count = Math.min(Math.max(1, Number(req.tail_rows || 5)), Math.max(0, lastRow - 1));
  const tailRows = count > 0
    ? sheet.getRange(lastRow - count + 1, 1, count, lastCol).getDisplayValues()
    : [];
  return jsonResponse({ ok: true, headers: headers, tail_rows: tailRows, last_row: lastRow });
}

function handleAppendRow(req) {
  const ss = SpreadsheetApp.openById(req.source_id);
  const sheet = req.sheet_name ? ss.getSheetByName(req.sheet_name) : ss.getSheets()[0];
  if (!sheet) {
    return jsonResponse({ ok: false, error: "Sheet not found: " + req.sheet_name });
  }
  const row = req.row;
  if (!Array.isArray(row) || row.length === 0) {
    return jsonResponse({ ok: false, error: "Missing non-empty row array." });
  }
  const lastRow = sheet.getLastRow();
  if (req.date && lastRow >= 2) {
    const existingDate = String(sheet.getRange(lastRow, 1).getDisplayValue()).trim();
    if (existingDate === String(req.date).trim()) {
      return jsonResponse({
        ok: true,
        skipped: true,
        reason: "Row for date " + existingDate + " already exists.",
        last_row: lastRow
      });
    }
  }
  sheet.appendRow(row);
  SpreadsheetApp.flush();
  return jsonResponse({ ok: true, appended: true, last_row: sheet.getLastRow() });
}

function handleFetch(req) {
  const ss = SpreadsheetApp.openById(req.source_id);
  const sheet = req.sheet_name ? ss.getSheetByName(req.sheet_name) : ss.getSheets()[0];
  if (!sheet) {
    return jsonResponse({ ok: false, error: "Sheet not found: " + req.sheet_name });
  }
  const lastRow = sheet.getLastRow();
  const lastCol = sheet.getLastColumn();
  if (lastRow === 0 || lastCol === 0) {
    return jsonResponse({ ok: true, rows: [], total_rows: 0 });
  }
  const startRow = Math.max(1, Number(req.start_row || 1));
  if (startRow > lastRow) {
    return jsonResponse({ ok: true, rows: [], total_rows: lastRow });
  }
  const numRows = Math.min(Number(req.num_rows || 2500), lastRow - startRow + 1);
  const rows = sheet.getRange(startRow, 1, numRows, lastCol).getDisplayValues();
  return jsonResponse({ ok: true, rows: rows, total_rows: lastRow });
}

function ensureSheetSize(sheet, reqRows, reqCols) {
  const curRows = sheet.getMaxRows();
  const curCols = sheet.getMaxColumns();
  if (reqRows > curRows) {
    sheet.insertRowsAfter(curRows, reqRows - curRows);
  }
  if (reqCols > curCols) {
    sheet.insertColumnsAfter(curCols, reqCols - curCols);
  }
}

function handlePrepareTab(req) {
  const ss = SpreadsheetApp.openById(req.dest_id);
  let sheet = ss.getSheetByName(req.tab_name);
  if (!sheet && req.rename_sheet1) {
    const sheet1 = ss.getSheetByName("Sheet1");
    if (sheet1) {
      sheet1.setName(req.tab_name);
      sheet = sheet1;
    }
  }
  if (!sheet) {
    sheet = ss.insertSheet(req.tab_name);
  }
  sheet.clear();
  ensureSheetSize(sheet, Number(req.rows || 1), Number(req.cols || 1));
  SpreadsheetApp.flush();
  return jsonResponse({ ok: true });
}

function handleWriteChunk(req) {
  const ss = SpreadsheetApp.openById(req.dest_id);
  const sheet = ss.getSheetByName(req.tab_name);
  if (!sheet) {
    return jsonResponse({ ok: false, error: "Destination tab not found: " + req.tab_name });
  }
  const values = req.values || [];
  if (values.length > 0 && values[0].length > 0) {
    const startRow = Number(req.start_row || 1);
    const numRows = values.length;
    const numCols = values[0].length;
    ensureSheetSize(sheet, startRow + numRows - 1, numCols);
    sheet.getRange(startRow, 1, numRows, numCols).setValues(values);
  }
  SpreadsheetApp.flush();
  return jsonResponse({ ok: true, written: values.length });
}

function handleFormatTab(req) {
  const ss = SpreadsheetApp.openById(req.dest_id);
  const sheet = ss.getSheetByName(req.tab_name);
  if (!sheet) {
    return jsonResponse({ ok: false, error: "Destination tab not found: " + req.tab_name });
  }
  const nCols = Math.max(1, Number(req.n_cols || sheet.getLastColumn() || 1));
  sheet
    .getRange(1, 1, 1, nCols)
    .setBackground("#1F4E78")
    .setFontColor("#FFFFFF")
    .setFontWeight("bold");

  sheet.setFrozenRows(1);
  sheet.setFrozenColumns(req.freeze_col_a ? 1 : 0);

  const sectionRows = req.section_rows || [];
  for (let i = 0; i < sectionRows.length; i++) {
    sheet
      .getRange(sectionRows[i], 1, 1, nCols)
      .setBackground("#1F4E78")
      .setFontColor("#FFFFFF")
      .setFontWeight("bold");
  }

  const subheaderRows = req.subheader_rows || [];
  for (let i = 0; i < subheaderRows.length; i++) {
    sheet
      .getRange(subheaderRows[i], 1, 1, nCols)
      .setBackground("#D9E1F2")
      .setFontColor("#000000")
      .setFontWeight("bold");
  }

  SpreadsheetApp.flush();
  return jsonResponse({ ok: true });
}

function handleApplyColorScales(req) {
  const ss = SpreadsheetApp.openById(req.dest_id);
  const rules = req.rules || [];
  const byTab = {};
  for (let i = 0; i < rules.length; i++) {
    const r = rules[i];
    if (!byTab[r.tab]) byTab[r.tab] = [];
    byTab[r.tab].push(r);
  }
  const tabs = Object.keys(byTab);
  for (let t = 0; t < tabs.length; t++) {
    const sheet = ss.getSheetByName(tabs[t]);
    if (!sheet) continue;
    const cfRules = [];
    const list = byTab[tabs[t]];
    for (let i = 0; i < list.length; i++) {
      const sp = list[i];
      const rng = sheet.getRange(
        sp.start_row + 1,
        sp.start_col + 1,
        sp.end_row - sp.start_row,
        sp.end_col - sp.start_col
      );
      let b = SpreadsheetApp.newConditionalFormatRule()
        .setGradientMinpointWithValue(sp.min_hex, SpreadsheetApp.InterpolationType.NUMBER, String(sp.min_val))
        .setGradientMaxpointWithValue(sp.max_hex, SpreadsheetApp.InterpolationType.NUMBER, String(sp.max_val))
        .setRanges([rng]);
      if (sp.mid_val !== null && sp.mid_val !== undefined && sp.mid_hex) {
        b = b.setGradientMidpointWithValue(sp.mid_hex, SpreadsheetApp.InterpolationType.NUMBER, String(sp.mid_val));
      }
      cfRules.push(b.build());
    }
    sheet.setConditionalFormatRules(cfRules);
  }
  SpreadsheetApp.flush();
  return jsonResponse({ ok: true, rules_applied: rules.length });
}

function handleRebuildCharts(req) {
  const ss = SpreadsheetApp.openById(req.dest_id);
  const specs = req.charts || [];
  const cleared = {};
  for (let i = 0; i < specs.length; i++) {
    const sp = specs[i];
    const target = ss.getSheetByName(sp.target_tab);
    if (!target) continue;
    if (!cleared[sp.target_tab]) {
      const existing = target.getCharts();
      for (let c = 0; c < existing.length; c++) {
        target.removeChart(existing[c]);
      }
      cleared[sp.target_tab] = true;
    }
    const domSheet = ss.getSheetByName(sp.domain.tab);
    if (!domSheet) continue;
    const typeMap = {
      LINE: Charts.ChartType.LINE,
      COLUMN: Charts.ChartType.COLUMN,
      BAR: Charts.ChartType.BAR,
      COMBO: Charts.ChartType.COMBO,
      AREA: Charts.ChartType.AREA
    };
    const chartType = typeMap[sp.chart_type] || Charts.ChartType.LINE;
    let builder = target.newChart()
      .setChartType(chartType)
      .setPosition(sp.anchor_row + 1, sp.anchor_col + 1, 8, 6)
      .setOption("title", sp.title)
      .setOption("width", sp.width || 660)
      .setOption("height", sp.height || 380)
      .setOption("legend", { position: "bottom" })
      .setNumHeaders(1);

    if (sp.x_title) builder = builder.setOption("hAxis.title", sp.x_title);
    if (sp.y_left_title) builder = builder.setOption("vAxes.0.title", sp.y_left_title);
    if (sp.y_right_title) builder = builder.setOption("vAxes.1.title", sp.y_right_title);
    if (sp.interpolate_nulls === false) builder = builder.setOption("interpolateNulls", false);

    builder = builder.addRange(
      domSheet.getRange(
        sp.domain.sr + 1,
        sp.domain.sc + 1,
        sp.domain.er - sp.domain.sr,
        sp.domain.ec - sp.domain.sc
      )
    );
    const seriesOpts = {};
    for (let s = 0; s < (sp.series || []).length; s++) {
      const ser = sp.series[s];
      const sSheet = ss.getSheetByName(ser.tab);
      if (!sSheet) continue;
      builder = builder.addRange(
        sSheet.getRange(ser.sr + 1, ser.sc + 1, ser.er - ser.sr, ser.ec - ser.sc)
      );
      const opt = { color: ser.color || "#1F4E78", targetAxisIndex: ser.axis === "RIGHT_AXIS" ? 1 : 0 };
      if (sp.chart_type === "COMBO" && ser.type) {
        opt.type = ser.type === "COLUMN" ? "bars" : "line";
      }
      if (ser.line_width) opt.lineWidth = ser.line_width;
      if (ser.dashed) opt.lineDashStyle = [6, 4];
      seriesOpts[s] = opt;
    }
    builder = builder.setOption("series", seriesOpts);
    target.insertChart(builder.build());
  }
  SpreadsheetApp.flush();
  return jsonResponse({ ok: true, charts_created: specs.length });
}

function findFolderByIdOrName(folderId, folderName, parentFolder) {
  if (folderId) {
    try {
      const f = DriveApp.getFolderById(folderId);
      if (f && !f.isTrashed()) return f;
    } catch (e) {
      // Fall back to name lookup
    }
  }
  if (parentFolder) {
    const it = parentFolder.getFoldersByName(folderName);
    if (it.hasNext()) return it.next();
    return null;
  }
  const it = DriveApp.getFoldersByName(folderName);
  if (it.hasNext()) return it.next();
  return null;
}

function findFileIdByNameInFolder(folder, fileName) {
  const it = folder.getFilesByName(fileName);
  while (it.hasNext()) {
    const f = it.next();
    if (!f.isTrashed()) return f.getId();
  }
  return "";
}

function upsertHtmlGoogleDoc(existingFileId, title, parentFolderId, htmlContent) {
  const token = ScriptApp.getOAuthToken();
  const boundary = "-------agent4_doc_boundary_" + Date.now();
  const metadata = {
    name: title,
    mimeType: "application/vnd.google-apps.document"
  };
  if (!existingFileId && parentFolderId) {
    metadata.parents = [parentFolderId];
  }
  const body =
    "--" + boundary + "\r\n" +
    "Content-Type: application/json; charset=UTF-8\r\n\r\n" +
    JSON.stringify(metadata) + "\r\n" +
    "--" + boundary + "\r\n" +
    "Content-Type: text/html; charset=UTF-8\r\n\r\n" +
    htmlContent + "\r\n" +
    "--" + boundary + "--";

  const url = existingFileId
    ? "https://www.googleapis.com/upload/drive/v3/files/" + encodeURIComponent(existingFileId) + "?uploadType=multipart&fields=id,name,webViewLink"
    : "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,webViewLink";

  const resp = UrlFetchApp.fetch(url, {
    method: existingFileId ? "patch" : "post",
    contentType: "multipart/related; boundary=" + boundary,
    headers: { Authorization: "Bearer " + token },
    payload: Utilities.newBlob(body, "multipart/related; boundary=" + boundary).getBytes(),
    muteHttpExceptions: true
  });
  const code = resp.getResponseCode();
  const txt = resp.getContentText();
  if (code < 200 || code >= 300) {
    throw new Error("Drive v3 HTML-to-Doc upload failed (HTTP " + code + "): " + txt);
  }
  return JSON.parse(txt);
}

function handlePublishDoc(req) {
  const htmlContent = req.html_content || "";
  if (!htmlContent) {
    return jsonResponse({ ok: false, error: "Missing html_content." });
  }
  const folderName = req.folder_name || "Agent Folder";
  const masterTitle = req.master_title || "4. Market Dynamics & Macro Insights";
  const archiveFolderName = req.archive_folder_name || "4. Market Insights Archive";
  const archiveTitle = req.archive_title || (new Date().toISOString().slice(0, 10) + " - Market Dynamics & Macro Insights");

  let agentFolder = findFolderByIdOrName(req.folder_id || "", folderName, null);
  if (!agentFolder) {
    agentFolder = DriveApp.createFolder(folderName);
  }
  const agentFolderId = agentFolder.getId();

  // 1. Update existing master doc in-place (or create inside Agent Folder)
  let masterId = "";
  if (req.master_doc_id) {
    try {
      const mf = DriveApp.getFileById(req.master_doc_id);
      if (mf && !mf.isTrashed()) masterId = mf.getId();
    } catch (e) {
      masterId = "";
    }
  }
  if (!masterId) {
    masterId = findFileIdByNameInFolder(agentFolder, masterTitle);
  }
  const masterRes = upsertHtmlGoogleDoc(masterId, masterTitle, agentFolderId, htmlContent);

  // 2. Locate or create Archive subfolder and upsert today's dated Google Doc
  let archiveFolder = findFolderByIdOrName(req.archive_folder_id || "", archiveFolderName, agentFolder);
  if (!archiveFolder) {
    archiveFolder = agentFolder.createFolder(archiveFolderName);
  }
  const archiveFolderId = archiveFolder.getId();
  const existingArchId = findFileIdByNameInFolder(archiveFolder, archiveTitle);
  const archiveRes = upsertHtmlGoogleDoc(existingArchId, archiveTitle, archiveFolderId, htmlContent);

  return jsonResponse({
    ok: true,
    agent_folder_id: agentFolderId,
    master_doc_id: masterRes.id,
    master_doc_name: masterRes.name,
    master_doc_url: masterRes.webViewLink || ("https://docs.google.com/document/d/" + masterRes.id + "/edit"),
    archive_folder_id: archiveFolderId,
    archive_doc_id: archiveRes.id,
    archive_doc_name: archiveRes.name,
    archive_doc_url: archiveRes.webViewLink || ("https://docs.google.com/document/d/" + archiveRes.id + "/edit")
  });
}

/**
 * Run this function ONCE from the Apps Script editor to trigger OAuth authorization
 * for Google Drive + UrlFetchApp scopes required by Agent 4's publish_doc action.
 */
function authorizeDriveAccess() {
  const it = DriveApp.getFoldersByName("Agent Folder");
  const found = it.hasNext() ? it.next().getName() : "Not found (will create on run)";
  const probe = UrlFetchApp.fetch("https://www.googleapis.com/drive/v3/about?fields=user", {
    headers: { Authorization: "Bearer " + ScriptApp.getOAuthToken() },
    muteHttpExceptions: true
  });
  Logger.log("Authorized! Agent Folder: " + found + " | Drive v3 HTTP " + probe.getResponseCode());
}
