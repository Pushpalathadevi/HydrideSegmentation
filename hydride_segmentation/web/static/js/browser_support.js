(function () {
  "use strict";
  var ua = navigator.userAgent || "", browser = "other", version = 0, match;
  if ((match = ua.match(/Edg\/(\d+)/))) { browser = "Edge"; version = Number(match[1]); }
  else if ((match = ua.match(/Edge\/(\d+)/))) { browser = "Edge"; version = Number(match[1]); }
  else if ((match = ua.match(/Chrome\/(\d+)/))) { browser = "Chrome"; version = Number(match[1]); }
  else if ((match = ua.match(/Firefox\/(\d+)/))) { browser = "Firefox"; version = Number(match[1]); }
  var minimum = { Edge: 18, Chrome: 76, Firefox: 68 };
  if (browser !== "other" && version >= minimum[browser]) return;
  var notice = document.createElement("div"); notice.setAttribute("role", "alert");
  notice.style.cssText = "position:relative;z-index:9999;padding:10px 16px;background:#fff4d6;color:#5f4612;border-bottom:1px solid #e2c875;font:14px/1.4 system-ui,sans-serif;text-align:center";
  var text = browser === "other" ? "This browser is not in the tested support list." : browser + " " + version + " is older than the tested minimum (" + minimum[browser] + ").";
  notice.innerHTML = "<strong>Browser update recommended.</strong> " + text + " Some features may not work correctly. Please update to a recent Edge, Chrome, or Firefox release.";
  document.addEventListener("DOMContentLoaded", function () { document.body.insertBefore(notice, document.body.firstChild); });
}());
