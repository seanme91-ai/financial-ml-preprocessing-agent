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
    if (action === "prepare_tab") {
      return handlePrepareTab(req);
    }
    if (action === "write_chunk") {
      return handleWriteChunk(req);
    }
    if (action === "format_tab") {
      return handleFormatTab(req);
    }
    return jsonResponse({ ok: false, error: "Unknown action: " + action });
  } catch (err) {
    return jsonResponse({ ok: false, error: String(err && err.stack ? err.stack : err) });
  }
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
