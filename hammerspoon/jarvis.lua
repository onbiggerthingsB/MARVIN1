-- Option-Space: front the JARVIS window (or launch it) and send /wake.
local function readBearer()
  local f = io.open(os.getenv("HOME") .. "/jarvis/state/hook_bearer", "r")
  if not f then return nil end
  local t = f:read("*l"); f:close(); return t
end

hs.hotkey.bind({ "alt" }, "space", function()
  hs.execute(os.getenv("HOME") .. "/jarvis/bin/jarvis", true)
  local bearer = readBearer()
  if bearer then
    hs.http.asyncPost("http://127.0.0.1:7777/wake", "",
      { ["Authorization"] = "Bearer " .. bearer }, function() end)
  end
end)
